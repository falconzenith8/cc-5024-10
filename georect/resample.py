"""图像反采样层: 把一个有版本的变形场作用到源图, 得到版本化的重采样结果。

反向采样方法与边界政策与既有实现完全一致(见 warp.py 模块 docstring):
对每个输出像素的目标坐标 y, 阻尼牛顿法解 f(x)=y; 失败原因逐像素记录;
收敛但落在源图之外的像素为"越界", alpha=0, 不编造内容。

与旧实现的区别只在接口形式:
  - 输入是 DeformationField(带版本与 source_result_id), 不是裸模型;
  - 容差 tol 必须由调用方显式给出, 不用隐式默认值掩盖采样决策;
  - 返回 ResampleResult, 失败统计与原因类型明确, 且保留旧 dict 形状的
    `to_legacy_dict()` 供古图配准调用方兼容使用。
数值运算本身(camera/牛顿迭代/双线性采样)保持逐次相同。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import FIELD_VERSION, DeformationField
from .tps import TPSModel, affine_part, evaluate, jacobian

RESAMPLE_VERSION = "georect-resample-v1"

FAIL_OK = 0
FAIL_SINGULAR = 1
FAIL_NO_CONVERGE = 2
FAIL_NONFINITE = 3
FAIL_NAMES = {0: "ok", 1: "奇异(detJ≈0)", 2: "未收敛", 3: "非有限"}


class ResampleError(ValueError):
    """反采样输入错误(输出尺寸、范围、容差)。"""


def invert_map(model: TPSModel, Y, tol: float, max_iter: int = 30,
               chunk: int = 20000):
    """逐点求逆。返回 (X, info), info 含 converged/iters/reason 数组。"""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[None, :]
    m = Y.shape[0]
    X = np.full((m, 2), np.nan)
    iters = np.zeros(m, dtype=int)
    reason = np.full(m, FAIL_NO_CONVERGE, dtype=np.uint8)

    c0, C1 = affine_part(model)
    det_c1 = C1[0, 0] * C1[1, 1] - C1[0, 1] * C1[1, 0]
    if abs(det_c1) > 1e-12:
        C1inv = np.array([[C1[1, 1], -C1[0, 1]], [-C1[1, 0], C1[0, 0]]]) / det_c1
        x0_all = (Y - c0) @ C1inv
    else:  # 仿射部分退化时退化为控制点质心
        x0_all = np.broadcast_to(model.src.mean(0), (m, 2)).copy()

    src_span = float(np.linalg.norm(model.src.max(0) - model.src.min(0)))
    max_step = 0.5 * max(src_span, 1.0)

    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        y = Y[s:e]
        x = x0_all[s:e].copy()
        active = np.ones(e - s, dtype=bool)
        it = np.zeros(e - s, dtype=int)
        rsn = np.full(e - s, FAIL_NO_CONVERGE, dtype=np.uint8)

        for _ in range(max_iter):
            idx = np.nonzero(active)[0]
            if len(idx) == 0:
                break
            xa = x[idx]
            F = evaluate(model, xa)
            J = jacobian(model, xa)
            r = F - y[idx]
            rn = np.linalg.norm(r, axis=1)

            bad = ~np.isfinite(xa).all(1) | ~np.isfinite(F).all(1)
            if bad.any():
                gi = idx[bad]
                rsn[gi] = FAIL_NONFINITE
                active[gi] = False
            conv = (rn < tol) & ~bad
            if conv.any():
                gi = idx[conv]
                rsn[gi] = FAIL_OK
                active[gi] = False

            rem = ~(bad | conv)
            if not rem.any():
                continue
            idr = idx[rem]
            Jr = J[rem]
            rr = r[rem]
            det = Jr[:, 0, 0] * Jr[:, 1, 1] - Jr[:, 0, 1] * Jr[:, 1, 0]
            scale = (np.abs(Jr).reshape(len(Jr), -1).max(1)) ** 2
            sing = np.abs(det) <= 1e-14 * np.maximum(scale, 1e-300)
            if sing.any():
                gi = idr[sing]
                rsn[gi] = FAIL_SINGULAR
                active[gi] = False
            ok = ~sing
            if not ok.any():
                continue
            ido = idr[ok]
            Jo = Jr[ok]
            ro = rr[ok]
            det_o = det[ok]
            delta = np.empty_like(ro)
            delta[:, 0] = (Jo[:, 1, 1] * ro[:, 0] - Jo[:, 0, 1] * ro[:, 1]) / det_o
            delta[:, 1] = (-Jo[:, 1, 0] * ro[:, 0] + Jo[:, 0, 0] * ro[:, 1]) / det_o
            dn = np.linalg.norm(delta, axis=1)
            over = dn > max_step
            delta[over] *= (max_step / dn[over])[:, None]
            x[ido] = x[ido] - delta
            it[ido] += 1

        X[s:e] = x
        iters[s:e] = it
        reason[s:e] = rsn

    info = {"converged": reason == FAIL_OK, "iters": iters, "reason": reason}
    return X, info


def sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """在连续坐标处双线性采样。坐标约定: 像素 j 覆盖区间 [j, j+1),
    其中心位于 j+0.5 —— 与地标点击/世界坐标的连续约定一致。
    返回 (m,C) 与 valid 掩膜(坐标落在图像覆盖范围 [0,W]x[0,H] 内)。"""
    H, W = img.shape[:2]
    xs = np.asarray(xs, dtype=np.float64) - 0.5  # 转为中心在整数位置的采样坐标
    ys = np.asarray(ys, dtype=np.float64) - 0.5
    valid = (xs >= -0.5) & (xs <= W - 0.5) & (ys >= -0.5) & (ys <= H - 0.5)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, W - 1)
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = np.clip(xs - np.floor(xs), 0, 1)
    wy = np.clip(ys - np.floor(ys), 0, 1)
    imgf = img.astype(np.float64)
    Ia = imgf[y0, x0]
    Ib = imgf[y0, x1]
    Ic = imgf[y1, x0]
    Id = imgf[y1, x1]
    wa = ((1 - wx) * (1 - wy))[:, None]
    wb = (wx * (1 - wy))[:, None]
    wc = ((1 - wx) * wy)[:, None]
    wd = (wx * wy)[:, None]
    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out, valid


@dataclass(frozen=True)
class ResampleResult:
    """版本化重采样结果。rgba 为 (h,w,4) uint8, 无数据像素 alpha=0。"""

    version: str
    field_version: str
    source_result_id: str
    rgba: np.ndarray
    valid: np.ndarray
    outside: np.ndarray
    failed: np.ndarray
    reason: np.ndarray
    stats: dict
    tol: float

    def __post_init__(self):
        if self.version != RESAMPLE_VERSION:
            raise ResampleError(f"重采样结果版本不匹配: {self.version!r}")
        if self.field_version != FIELD_VERSION:
            raise ResampleError(f"变形场版本不匹配: {self.field_version!r}")
        if not isinstance(self.source_result_id, str) or not self.source_result_id:
            raise ResampleError("重采样结果必须记录 source_result_id")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ResampleError("tol 必须为正的有限数")

    def to_legacy_dict(self) -> dict:
        """旧入口返回的 dict 形状, 供既有调用方无改动读取。"""
        return {
            "rgba": self.rgba,
            "valid": self.valid,
            "outside": self.outside,
            "failed": self.failed,
            "reason": self.reason,
            "stats": self.stats,
        }


def resample_field(field: DeformationField, img: np.ndarray, out_extent,
                   out_w: int, out_h: int, *, tol: float,
                   max_iter: int = 30) -> ResampleResult:
    """把版本化变形场作用到源图。tol 必须显式给出(输出平面坐标单位)。"""
    if not isinstance(field, DeformationField):
        raise TypeError("resample_field 需要 DeformationField")
    img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ResampleError("源图应为 (H,W) 或 (H,W,C)")
    out_w, out_h = int(out_w), int(out_h)
    if not (16 <= out_w <= 8000 and 16 <= out_h <= 8000):
        raise ResampleError("输出尺寸需在 16..8000 像素之间")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ResampleError("tol 必须为正的有限数")

    xmin, ymin, xmax, ymax = (float(v) for v in out_extent)
    if not (np.isfinite([xmin, ymin, xmax, ymax]).all() and xmax > xmin and ymax > ymin):
        raise ResampleError(f"无效的目标范围: {out_extent!r}")

    px = (xmax - xmin) / out_w
    py = (ymax - ymin) / out_h
    cols = xmin + (np.arange(out_w) + 0.5) * px
    rows = ymax - (np.arange(out_h) + 0.5) * py
    gx, gy = np.meshgrid(cols, rows)
    Y = np.column_stack([gx.ravel(), gy.ravel()])

    X, info = invert_map(field.model, Y, tol=tol, max_iter=max_iter)

    sx, sy = X[:, 0], X[:, 1]
    vals, in_img = sample_bilinear(img, np.nan_to_num(sx), np.nan_to_num(sy))
    conv = info["converged"]
    outside = conv & ~in_img
    failed = ~conv

    rgba = np.zeros((out_h * out_w, 4), dtype=np.uint8)
    valid = conv & in_img
    rgba[valid, :3] = np.clip(vals[valid] + 0.5, 0, 255).astype(np.uint8)
    rgba[valid, 3] = 255

    stats = {
        "tol": float(tol),
        "max_iter": max_iter,
        "n_total": int(len(Y)),
        "n_valid": int(valid.sum()),
        "n_outside": int(outside.sum()),
        "n_failed": int(failed.sum()),
        "fail_singular": int((info["reason"] == FAIL_SINGULAR).sum()),
        "fail_no_converge": int((info["reason"] == FAIL_NO_CONVERGE).sum()),
        "fail_nonfinite": int((info["reason"] == FAIL_NONFINITE).sum()),
        "mean_iters": float(info["iters"][conv].mean()) if conv.any() else 0.0,
    }
    return ResampleResult(
        version=RESAMPLE_VERSION,
        field_version=field.version,
        source_result_id=field.source_result_id,
        rgba=rgba.reshape(out_h, out_w, 4),
        valid=valid.reshape(out_h, out_w),
        outside=outside.reshape(out_h, out_w),
        failed=failed.reshape(out_h, out_w),
        reason=info["reason"].reshape(out_h, out_w),
        stats=stats,
        tol=tol,
    )

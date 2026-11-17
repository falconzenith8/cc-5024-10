"""反向采样: 把源图重采样到目标网格。

求逆方法(显式说明):
  TPS 正映射 f: 源平面 -> 目标平面 没有解析逆。对每个输出像素的目标
  坐标 y, 用阻尼牛顿法解 f(x) = y:
    x_{k+1} = x_k - J(x_k)^{-1} (f(x_k) - y)
  - 初值: 仿射部分的逆 (y - c0)·C1^{-1}, 通常已很接近;
  - 步长限制(信赖域): 单步位移不超过源域对角线的一半, 防止发散;
  - 收敛判据: |f(x) - y| < tol, tol 默认取 0.01 个输出像素;
  - 失败情形(逐像素记录原因):
      * |det J| 过小(局部奇异, 多发生在折叠区) -> "奇异";
      * 超过最大迭代次数未收敛 -> "未收敛";
      * 出现 NaN/Inf -> "非有限";
      * 收敛到的源坐标落在源图之外 -> "越界", 不采样、不凭空生成内容。
  失败/越界像素在输出 RGBA 中 alpha=0, 并汇总统计与掩膜返回。
"""
from __future__ import annotations

import numpy as np

from .tps import TPSModel, affine_part, evaluate, jacobian

FAIL_OK = 0
FAIL_SINGULAR = 1
FAIL_NO_CONVERGE = 2
FAIL_NONFINITE = 3
FAIL_NAMES = {0: "ok", 1: "奇异(detJ≈0)", 2: "未收敛", 3: "非有限"}


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


def resample_to_grid(model: TPSModel, img: np.ndarray, out_extent,
                     out_w: int, out_h: int, tol: float | None = None,
                     max_iter: int = 30):
    """把源图 img (H,W,C) 重采样到目标网格。

    out_extent = (xmin, ymin, xmax, ymax) 目标平面范围; 输出第 0 行对应
    ymax(世界坐标 y 向上)。返回 RGBA uint8 与统计信息; 无数据像素 alpha=0。
    """
    xmin, ymin, xmax, ymax = out_extent
    px = (xmax - xmin) / out_w
    py = (ymax - ymin) / out_h
    if tol is None:
        tol = 0.01 * min(px, py)
    cols = xmin + (np.arange(out_w) + 0.5) * px
    rows = ymax - (np.arange(out_h) + 0.5) * py
    gx, gy = np.meshgrid(cols, rows)
    Y = np.column_stack([gx.ravel(), gy.ravel()])

    X, info = invert_map(model, Y, tol=tol, max_iter=max_iter)

    H, W = img.shape[:2]
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
    return {
        "rgba": rgba.reshape(out_h, out_w, 4),
        "valid": valid.reshape(out_h, out_w),
        "outside": outside.reshape(out_h, out_w),
        "failed": failed.reshape(out_h, out_w),
        "reason": info["reason"].reshape(out_h, out_w),
        "stats": stats,
    }

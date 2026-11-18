"""古地图影像配准工具 —— Flask 服务。

接口:
  POST /api/upload/<source|reference>  上传图像, 返回图像 id
  POST /api/fit                        求解: 诊断 + 不确定性 + 变形网格预览
  POST /api/warp                       图像反采样, 返回 RGBA PNG(base64) 与统计

拟合与重采样共用同一个版本化求解结果(georect.kernel.solve_registration):
同一组地标/λ/范围得到同一 result_id, 预览、叠图与导出因此读取同一结果。
"""
from __future__ import annotations

import base64
import io
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from georect.export import solve_to_dict
from georect.kernel import (LandmarkConstraintError, LandmarkConstraints,
                            RegistrationError, RegistrationFailure,
                            solve_registration)
from georect.resample import ResampleError, resample_field
from georect.tps import evaluate
from georect.uncertainty import sigma_hat_from_loocv, uncertainty_grid

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

_IMAGES: dict[str, np.ndarray] = {}

# 图像反采样的容差政策(与既有行为一致): 最小输出像素的 0.01。
# 调用点显式算出并传入, 库函数不再用默认值替代这个决策。
TOL_PIXEL_FRACTION = 0.01


def _err(msg, code=400, extra=None):
    body = {"ok": False, "errors": [str(msg)]}
    if extra:
        body.update(extra)
    return jsonify(body), code


def _json_safe(obj):
    """把非有限浮点转为 None: NaN/Inf 不是合法 JSON, 不能发给前端。"""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def _solve_payload():
    """解析并校验地标约束; 失败时返回 (None, 错误响应)。"""
    d = request.get_json(force=True, silent=True)
    if not isinstance(d, dict):
        return None, _err("请求体必须为 JSON 对象")
    try:
        src = np.asarray(d["source_points"], float)
        dst = np.asarray(d["target_points"], float)
        lam = float(d.get("lambda", 0.0))
        target_extent = tuple(float(v) for v in d["target_extent"])
        if len(target_extent) != 4:
            raise ValueError("target_extent 需要 4 个数值")
        src_extent = d.get("src_extent")
        if src_extent is not None:
            src_extent = tuple(float(v) for v in src_extent)
            if len(src_extent) != 4:
                raise ValueError("src_extent 需要 4 个数值")
    except (KeyError, TypeError, ValueError) as e:
        return None, _err(f"参数错误: {e}")
    try:
        constraints = LandmarkConstraints(src=src, dst=dst, lam=lam)
    except LandmarkConstraintError as e:
        return None, _err(str(e))
    return {"payload": d, "constraints": constraints,
            "target_extent": target_extent, "src_extent": src_extent}, None


def _solve(constraints, src_extent):
    return solve_registration(constraints, src_extent=src_extent)


def _tolerance(extent, out_w, out_h):
    xmin, ymin, xmax, ymax = extent
    px = (xmax - xmin) / int(out_w)
    py = (ymax - ymin) / int(out_h)
    return TOL_PIXEL_FRACTION * min(px, py)


def _ensure_same_result(resp_result: str, expected_id: str, what: str):
    """防御性检查: 两条路径拿到的必须是同一个版本化结果。"""
    if resp_result != expected_id:
        raise RegistrationError(
            f"{what} 使用的求解结果({resp_result})与请求的({expected_id})不一致")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/upload/<kind>")
def upload(kind):
    if kind not in ("source", "reference"):
        return _err("kind 必须为 source 或 reference")
    f = request.files.get("file")
    if f is None:
        return _err("缺少文件")
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return _err(f"无法解析图像: {e}")
    image_id = uuid.uuid4().hex
    _IMAGES[image_id] = np.asarray(img)
    return jsonify({"ok": True, "id": image_id, "width": img.width, "height": img.height})


def _fit_response(result, target_extent, src_extent, unc_nx, unc_ny):
    """把求解结果转换为 /api/fit 的响应体(既有字段保持不变)。"""
    if isinstance(result, RegistrationFailure):
        return {
            "ok": False,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "solve": {"version": result.version, "result_id": result.result_id,
                      "ok": False},
            "duplicates_src": result.diagnostics.get("duplicates_src", []),
            "duplicates_dst": result.diagnostics.get("duplicates_dst", []),
            "contradictions": result.diagnostics.get("contradictions", []),
            "geometry": result.diagnostics.get("geometry"),
            "condition_number": result.diagnostics.get("condition_number"),
        }

    diag = result.diagnostics
    field = result.field()
    model = result.model
    constraints = result.constraints
    sigma = sigma_hat_from_loocv(constraints.src, constraints.dst, constraints.lam)
    unc = uncertainty_grid(model, constraints.dst, sigma, target_extent,
                           nx=unc_nx, ny=unc_ny)

    folds = diag.get("folds")
    fold_pts_target = []
    if folds and folds["fold_cells_src"]:
        fold_pts_target = evaluate(model, np.asarray(folds["fold_cells_src"])).tolist()

    grid_extent = src_extent if src_extent else (
        float(constraints.src[:, 0].min()), float(constraints.src[:, 1].min()),
        float(constraints.src[:, 0].max()), float(constraints.src[:, 1].max()))
    loocv = diag["loocv_residuals"]
    # LOOCV 无定义(地标太少/留一后系统奇异)时给 None, 不用 0 冒充"误差为零"
    loocv_json = np.where(np.isfinite(loocv), loocv, np.nan).tolist()

    resp = {
        "ok": True,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "solve": {"version": result.version, "result_id": result.result_id,
                  "ok": True},
        "model": solve_to_dict(result),
        "fitted": evaluate(model, constraints.src).tolist(),
        "fit_residuals": diag["fit_residuals"].tolist(),
        "loocv": loocv_json,
        "loocv_flags": diag["loocv_flags"].tolist(),
        "rms_error": diag["rms_error"],
        "max_error": diag["max_error"],
        "loocv_rms": diag["loocv_rms"],
        "sigma_hat": sigma,
        "bending_energy": field.bending_energy(),
        "folds": None if folds is None else {
            "n_fold": folds["n_fold"],
            "frac_fold": folds["frac_fold"],
            "min_det": folds["min_det"],
            "finite": folds["finite"],
            "fold_points_target": fold_pts_target,
        },
        "uncertainty": {
            "xs": unc["xs"].tolist(),
            "ys": unc["ys"].tolist(),
            "std": unc["std"].tolist(),
            "inside_hull": unc["inside_hull"].tolist(),
            "sigma": sigma,
        },
        "hull": unc["hull"].tolist(),
        "grid_lines": field.grid_lines(grid_extent),
        "duplicates_src": diag.get("duplicates_src", []),
        "duplicates_dst": diag.get("duplicates_dst", []),
        "contradictions": diag.get("contradictions", []),
        "geometry": diag.get("geometry"),
        "condition_number": diag.get("condition_number"),
    }
    return resp


@app.post("/api/fit")
def fit():
    parsed, err = _solve_payload()
    if err:
        return err
    try:
        result = _solve(parsed["constraints"], parsed["src_extent"])
        resp = _fit_response(
            result, parsed["target_extent"], parsed["src_extent"],
            int(parsed["payload"].get("unc_nx", 90)),
            int(parsed["payload"].get("unc_ny", 90)))
    except (RegistrationError, ValueError, TypeError) as e:
        return _err(f"求解失败: {e}", 400)
    except Exception as e:  # noqa: BLE001
        app.logger.exception("unexpected /api/fit failure")
        return _err(f"内部错误: {e}", 500)
    return jsonify(_json_safe(resp))


def _downsample_mask(mask, max_w=160):
    h, w = mask.shape
    if w <= max_w:
        return mask.astype(bool).tolist()
    bw = int(np.ceil(w / max_w))
    bh = bw
    ph = (-h) % bh
    pw = (-w) % bw
    padded = np.pad(mask, ((0, ph), (0, pw)), constant_values=False)
    nh, nw = padded.shape
    small = padded.reshape(nh // bh, bh, nw // bw, bw).any(axis=(1, 3))
    return small.astype(bool).tolist()


@app.post("/api/warp")
def warp():
    parsed, err = _solve_payload()
    if err:
        return err
    payload = parsed["payload"]
    try:
        img = _IMAGES[payload["source_id"]]
    except (KeyError, TypeError):
        return _err("未知的 source_id, 请重新上传扫描图")
    out_w = int(payload["out_width"])
    out_h = int(payload["out_height"])
    if not (16 <= out_w <= 8000 and 16 <= out_h <= 8000):
        return _err("输出尺寸需在 16..8000 像素之间")

    src_extent = (0.0, 0.0, float(img.shape[1]), float(img.shape[0]))
    try:
        result = _solve(parsed["constraints"], src_extent)
        if isinstance(result, RegistrationFailure):
            return jsonify({
                "ok": False,
                "errors": list(result.errors),
                "warnings": list(result.warnings),
                "solve": {"version": result.version,
                          "result_id": result.result_id, "ok": False},
            })
        field = result.field()
        tol = _tolerance(parsed["target_extent"], out_w, out_h)
        out = resample_field(field, img, parsed["target_extent"], out_w, out_h, tol=tol)
        _ensure_same_result(out.source_result_id, result.result_id, "重采样")
    except (RegistrationError, ResampleError) as e:
        return _err(f"重采样失败: {e}", 400)
    except Exception as e:  # noqa: BLE001
        app.logger.exception("unexpected /api/warp failure")
        return _err(f"内部错误: {e}", 500)

    png = Image.fromarray(out.rgba, "RGBA")
    buf = io.BytesIO()
    png.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return jsonify({
        "ok": True,
        "solve": {"version": result.version, "result_id": result.result_id,
                  "ok": True},
        "resample": {"version": out.version,
                     "source_result_id": out.source_result_id,
                     "tol": out.tol},
        "png_base64": b64,
        "stats": out.stats,
        "failed_mask": _downsample_mask(out.failed),
        "outside_mask": _downsample_mask(out.outside),
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

"""古地图影像配准工具 —— Flask 服务。

接口:
  POST /api/upload/<source|reference>  上传图像, 返回图像 id
  POST /api/fit                        拟合 TPS + 诊断 + 不确定性 + 变形网格
  POST /api/warp                       反向采样, 返回 RGBA PNG(base64) 与统计
静态文件即前端界面。
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

from georect.diagnostics import run_all
from georect.export import model_to_dict
from georect.tps import bending_energy, evaluate
from georect.uncertainty import sigma_hat_from_loocv, uncertainty_grid
from georect.warp import resample_to_grid

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

_IMAGES: dict[str, np.ndarray] = {}


def _err(msg, code=400):
    return jsonify({"ok": False, "errors": [str(msg)]}), code


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


def _deformed_grid_lines(model, extent, n_lines=16, n_pts=80):
    """源域等距网格线映射到目标域, 用于展示变形(含翻折可见性)。"""
    xmin, ymin, xmax, ymax = extent
    lines = []
    for i in range(n_lines + 1):
        x = xmin + (xmax - xmin) * i / n_lines
        ys = np.linspace(ymin, ymax, n_pts)
        pts = np.column_stack([np.full(n_pts, x), ys])
        lines.append(evaluate(model, pts).tolist())
    for j in range(n_lines + 1):
        y = ymin + (ymax - ymin) * j / n_lines
        xs = np.linspace(xmin, xmax, n_pts)
        pts = np.column_stack([xs, np.full(n_pts, y)])
        lines.append(evaluate(model, pts).tolist())
    return lines


@app.post("/api/fit")
def fit():
    d = request.get_json(force=True)
    try:
        src = np.asarray(d["source_points"], float)
        dst = np.asarray(d["target_points"], float)
        lam = float(d.get("lambda", 0.0))
        tgt_extent = d["target_extent"]
        src_extent = d.get("src_extent")
    except (KeyError, TypeError, ValueError) as e:
        return _err(f"参数错误: {e}")

    diag = run_all(src, dst, lam, src_extent=src_extent)
    resp = {
        "ok": "model" in diag,
        "errors": diag["errors"],
        "warnings": diag["warnings"],
        "duplicates_src": diag.get("duplicates_src", []),
        "duplicates_dst": diag.get("duplicates_dst", []),
        "contradictions": diag.get("contradictions", []),
        "geometry": diag.get("geometry"),
        "condition_number": diag.get("condition_number"),
    }
    if "model" not in diag:
        return jsonify(resp)

    model = diag["model"]
    sigma = sigma_hat_from_loocv(src, dst, lam)
    unc = uncertainty_grid(model, dst, sigma, tuple(map(float, tgt_extent)),
                           nx=int(d.get("unc_nx", 90)), ny=int(d.get("unc_ny", 90)))

    folds = diag.get("folds")
    fold_pts_target = []
    if folds and folds["fold_cells_src"]:
        fold_pts_target = evaluate(model, np.asarray(folds["fold_cells_src"])).tolist()

    grid_extent = src_extent if src_extent else (
        float(src[:, 0].min()), float(src[:, 1].min()),
        float(src[:, 0].max()), float(src[:, 1].max()))

    resp.update({
        "model": model_to_dict(model, dst),
        "fitted": evaluate(model, src).tolist(),
        "fit_residuals": diag["fit_residuals"].tolist(),
        "loocv": np.nan_to_num(diag["loocv_residuals"]).tolist(),
        "loocv_flags": diag["loocv_flags"].tolist(),
        "rms_error": diag["rms_error"],
        "max_error": diag["max_error"],
        "loocv_rms": diag["loocv_rms"],
        "sigma_hat": sigma,
        "bending_energy": bending_energy(model),
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
        "grid_lines": _deformed_grid_lines(model, grid_extent),
    })
    return jsonify(resp)


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
    d = request.get_json(force=True)
    try:
        img = _IMAGES[d["source_id"]]
        src = np.asarray(d["source_points"], float)
        dst = np.asarray(d["target_points"], float)
        lam = float(d.get("lambda", 0.0))
        extent = tuple(map(float, d["target_extent"]))
        out_w = int(d["out_width"])
        out_h = int(d["out_height"])
    except (KeyError, TypeError, ValueError) as e:
        return _err(f"参数错误: {e}")
    if not (16 <= out_w <= 8000 and 16 <= out_h <= 8000):
        return _err("输出尺寸需在 16..8000 像素之间")

    diag = run_all(src, dst, lam, src_extent=(0, 0, img.shape[1], img.shape[0]))
    if "model" not in diag:
        return jsonify({"ok": False, "errors": diag["errors"]})

    out = resample_to_grid(diag["model"], img, extent, out_w, out_h)
    png = Image.fromarray(out["rgba"], "RGBA")
    buf = io.BytesIO()
    png.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return jsonify({
        "ok": True,
        "png_base64": b64,
        "stats": out["stats"],
        "failed_mask": _downsample_mask(out["failed"]),
        "outside_mask": _downsample_mask(out["outside"]),
        "errors": diag["errors"],
        "warnings": diag["warnings"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

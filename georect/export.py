"""变形参数的导出/导入 (JSON), 保证结果可复现、可交换。"""
from __future__ import annotations

import datetime

import numpy as np

from .tps import KERNEL_NAME, TPSModel

FORMAT_VERSION = "1.0"


def model_to_dict(model: TPSModel, dst, extra=None) -> dict:
    d = {
        "format": "georect-tps-params",
        "version": FORMAT_VERSION,
        "kernel": KERNEL_NAME,
        "lambda": model.lam,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_points": np.asarray(model.src, float).tolist(),
        "target_points": np.asarray(dst, float).tolist(),
        "weights": np.asarray(model.W, float).tolist(),
        "affine": np.asarray(model.C, float).tolist(),
        "normalize": {"center": np.asarray(model.center, float).tolist(),
                      "scale": float(model.scale)},
    }
    if extra:
        d.update(extra)
    return d


def model_from_dict(d: dict) -> TPSModel:
    if d.get("kernel") != KERNEL_NAME:
        raise ValueError(f"不支持的核: {d.get('kernel')}")
    norm = d.get("normalize") or {}
    return TPSModel(
        src=np.asarray(d["source_points"], float),
        W=np.asarray(d["weights"], float),
        C=np.asarray(d["affine"], float),
        lam=float(d["lambda"]),
        center=np.asarray(norm.get("center", [0.0, 0.0]), float),
        scale=float(norm.get("scale", 1.0)),
    )

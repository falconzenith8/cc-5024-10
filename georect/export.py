"""变形参数的导出/导入 (JSON), 保证结果可复现、可交换。

导出读取的是同一个版本化求解结果对象(kernel.RegistrationSolve):
预览、重采样与导出不各自重新拟合, 因而不会给出互相矛盾的结果。
model_to_dict 的字段保持既有格式, 仅在 results 版本化时追加求解元数据。
"""
from __future__ import annotations

import datetime

import numpy as np

from .kernel import SOLVE_VERSION, RegistrationSolve
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


def solve_to_dict(result: RegistrationSolve, extra=None) -> dict:
    """把版本化求解结果导出为可交换 JSON(字段与旧格式向前兼容)。"""
    if not isinstance(result, RegistrationSolve) or not result.ok:
        raise TypeError("solve_to_dict 需要成功的 RegistrationSolve")
    if result.version != SOLVE_VERSION:
        raise ValueError(f"不支持的求解结果版本: {result.version!r}")
    d = model_to_dict(
        result.model, result.constraints.dst,
        extra=dict(extra or {}, solve={
            "version": result.version,
            "result_id": result.result_id,
        }),
    )
    # 版本化结果导出的参数版本标为 1.1; 既有字段与语义不变
    d["version"] = "1.1"
    return d

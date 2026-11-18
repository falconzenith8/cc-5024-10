"""变形求值层: 把已求解的 TPS 模型包装为有版本的变形场对象。

与"地标约束"(求解输入, kernel 层)和"图像反采样"(像素重采样, resample 层)
分开。本模块只负责在给定点集上求 f(x)、雅可比、位移网格与网格线;
不重新拟合、不接触图像像素。

数值原语(kernel / 线性求解 / 用户坐标求值)仍在 tps.py, 三层共用,
避免为拆分而复制公式、改变浮点运算次序。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .tps import (TPSModel, affine_part, bending_energy, evaluate, jacobian)

FIELD_VERSION = "georect-field-v1"


class FieldError(ValueError):
    """变形场构造/版本错误(编程错误, 不是用户数据问题)。"""


@dataclass(frozen=True)
class DisplacementFrame:
    """目标平面上的位移场采样: 源点 p 与其像 f(p)。

    points 的行优先顺序与 values 一一对应; row-major 展平自 (ny, nx),
    行 0 对应 ymin, 与 uncertainty_grid 的网格约定一致。
    """

    version: str
    extent: tuple
    nx: int
    ny: int
    xs: np.ndarray
    ys: np.ndarray
    points: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        if self.version != FIELD_VERSION:
            raise FieldError(f"位移场版本不匹配: {self.version!r}")
        if self.points.shape != self.values.shape or self.points.ndim != 2:
            raise FieldError("位移场 points/values 形状不一致")
        if self.points.shape[0] != self.nx * self.ny:
            raise FieldError("位移场点数与 nx*ny 不一致")


@dataclass(frozen=True)
class DeformationField:
    """有版本的独立变形场对象。

    source_result_id 记录它来自哪次求解结果; 版本与来源都是必填项,
    不允许用默认值把"无来源/旧版本"掩盖过去。
    """

    version: str
    model: TPSModel
    source_result_id: str

    def __post_init__(self):
        if self.version != FIELD_VERSION:
            raise FieldError(f"变形场版本不匹配: {self.version!r}")
        if not isinstance(self.source_result_id, str) or not self.source_result_id:
            raise FieldError("变形场必须记录 source_result_id")
        if self.model.n < 3:
            raise FieldError("变形场控制点不足")

    def evaluate(self, X) -> np.ndarray:
        """在点集 X (m,2) 上求变形 f(X)。"""
        return evaluate(self.model, X)

    def jacobian(self, X) -> np.ndarray:
        return jacobian(self.model, X)

    def bending_energy(self) -> float:
        return bending_energy(self.model)

    def affine_part(self):
        return affine_part(self.model)

    def displacement(self, extent, nx: int, ny: int) -> DisplacementFrame:
        """在目标平面矩形上采样位移场。extent=(xmin,ymin,xmax,ymax)。"""
        xmin, ymin, xmax, ymax = (float(v) for v in extent)
        if not (np.isfinite([xmin, ymin, xmax, ymax]).all() and xmax > xmin and ymax > ymin):
            raise FieldError(f"无效的位移场范围: {extent!r}")
        if nx < 2 or ny < 2:
            raise FieldError("位移场网格每维至少 2 个采样点")
        xs = np.linspace(xmin, xmax, int(nx))
        ys = np.linspace(ymin, ymax, int(ny))
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        vals = self.evaluate(pts)
        return DisplacementFrame(
            version=FIELD_VERSION, extent=(xmin, ymin, xmax, ymax),
            nx=int(nx), ny=int(ny), xs=xs, ys=ys, points=pts, values=vals)

    def grid_lines(self, extent, n_lines: int = 16, n_pts: int = 80):
        """源域等距网格线映射到目标域, 用于展示变形(含翻折可见性)。"""
        xmin, ymin, xmax, ymax = (float(v) for v in extent)
        if not (np.isfinite([xmin, ymin, xmax, ymax]).all() and xmax > xmin and ymax > ymin):
            raise FieldError(f"无效的网格线范围: {extent!r}")
        if n_lines < 1 or n_pts < 2:
            raise FieldError("网格线参数无效")
        lines = []
        for i in range(n_lines + 1):
            x = xmin + (xmax - xmin) * i / n_lines
            ys = np.linspace(ymin, ymax, n_pts)
            pts = np.column_stack([np.full(n_pts, x), ys])
            lines.append(self.evaluate(pts).tolist())
        for j in range(n_lines + 1):
            y = ymin + (ymax - ymin) * j / n_lines
            xs = np.linspace(xmin, xmax, n_pts)
            pts = np.column_stack([xs, np.full(n_pts, y)])
            lines.append(self.evaluate(pts).tolist())
        return lines


def model_fingerprint(model: TPSModel) -> str:
    """模型内容指纹, 供没有求解结果的旧模型显式标注来源。"""
    payload = {
        "kernel": "tps-r2logr",
        "src": np.asarray(model.src, float).tolist(),
        "W": np.asarray(model.W, float).tolist(),
        "C": np.asarray(model.C, float).tolist(),
        "lambda": float(model.lam),
        "center": np.asarray(model.center, float).tolist(),
        "scale": float(model.scale),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def field_from_model(model: TPSModel, source_result_id: str) -> DeformationField:
    """从模型构造变形场; 来源必须由调用方显式给出。"""
    return DeformationField(version=FIELD_VERSION, model=model,
                            source_result_id=source_result_id)


def legacy_field_from_model(model: TPSModel) -> DeformationField:
    """兼容适配: 旧入口只有模型、没有求解结果, 用内容指纹显式标注来源。"""
    return field_from_model(model, "legacy-model:" + model_fingerprint(model))

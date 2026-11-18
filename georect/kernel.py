"""配准内核: 地标约束 -> 版本化求解结果。

分层边界:
  - 地标约束 (LandmarkConstraints): 只有求解输入 (src, dst, λ),
    不携带任何模型、诊断或图像;
  - 求解结果 (RegistrationSolve / RegistrationFailure): 不可变、带版本号
    与确定性 result_id 的独立对象。成功与失败是两种明确的返回类型,
    调用方不需要靠 getattr/默认值猜测;
  - 变形求值 (georect.field) 与图像反采样 (georect.resample) 分开,
    只读取求解结果给出的模型。

旧入口 (diagnostics.run_all / tps.fit_tps / warp.resample_to_grid) 原样保留,
作为古图配准及其它既有调用的兼容适配; 本模块的求解路径复用同一组
数值原语 (run_all), 不改变拟合目标、正则化与边界政策。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .diagnostics import run_all
from .field import DeformationField, field_from_model
from .tps import TPSModel

SOLVE_VERSION = "georect-solve-v1"


class RegistrationError(ValueError):
    """配准内核的可预期错误基类。"""


class LandmarkConstraintError(RegistrationError):
    """地标约束本身非法(形状、点数、非有限值、λ)。"""


class RegistrationNotSolved(RegistrationError):
    """对失败结果请求变形场。"""


@dataclass(frozen=True)
class LandmarkConstraints:
    """地标约束: 求解的唯一输入。坐标均为用户坐标系。"""

    src: np.ndarray
    dst: np.ndarray
    lam: float

    def __post_init__(self):
        src = np.asarray(self.src, dtype=np.float64)
        dst = np.asarray(self.dst, dtype=np.float64)
        if src.ndim != 2 or src.shape[1] != 2:
            raise LandmarkConstraintError("src 应为 (n,2) 数组")
        if dst.shape != src.shape:
            raise LandmarkConstraintError("dst 形状应与 src 一致 (n,2)")
        if src.shape[0] < 3:
            raise LandmarkConstraintError("至少需要 3 个控制点才能拟合 TPS")
        if not (np.isfinite(src).all() and np.isfinite(dst).all()):
            raise LandmarkConstraintError("控制点坐标包含 NaN 或 Inf, 拒绝拟合")
        lam = float(self.lam)
        if not np.isfinite(lam) or lam < 0:
            raise LandmarkConstraintError("λ 必须为有限非负数")
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dst", dst)
        object.__setattr__(self, "lam", lam)

    @property
    def n(self) -> int:
        return int(self.src.shape[0])

    def fingerprint(self) -> str:
        payload = {
            "src": self.src.tolist(),
            "dst": self.dst.tolist(),
            "lambda": self.lam,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class RegistrationSolve:
    """成功的求解结果: 有版本、有确定性 id 的独立对象。"""

    version: str
    result_id: str
    constraints: LandmarkConstraints
    model: TPSModel
    diagnostics: dict
    errors: tuple
    warnings: tuple

    def __post_init__(self):
        if self.version != SOLVE_VERSION:
            raise RegistrationError(f"求解结果版本不匹配: {self.version!r}")
        if not isinstance(self.result_id, str) or not self.result_id:
            raise RegistrationError("求解结果必须带 result_id")
        if not isinstance(self.constraints, LandmarkConstraints):
            raise RegistrationError("求解结果必须带地标约束")
        if self.model.n != self.constraints.n:
            raise RegistrationError("模型控制点数与约束不一致")

    @property
    def ok(self) -> bool:
        return True

    def field(self) -> DeformationField:
        """构造与本次求解结果绑定的变形场。"""
        return field_from_model(self.model, self.result_id)


@dataclass(frozen=True)
class RegistrationFailure:
    """失败的求解结果: 明确失败类型, 附完整诊断与错误文本。"""

    version: str
    result_id: str
    constraints: LandmarkConstraints
    diagnostics: dict
    errors: tuple
    warnings: tuple

    def __post_init__(self):
        if self.version != SOLVE_VERSION:
            raise RegistrationError(f"求解结果版本不匹配: {self.version!r}")
        if not isinstance(self.result_id, str) or not self.result_id:
            raise RegistrationError("求解失败结果也必须带 result_id")
        if not self.errors:
            raise RegistrationError("求解失败结果必须包含至少一条错误")
        if not isinstance(self.constraints, LandmarkConstraints):
            raise RegistrationError("求解结果必须带地标约束")

    @property
    def ok(self) -> bool:
        return False

    def field(self) -> DeformationField:
        raise RegistrationNotSolved("; ".join(self.errors))


def _result_id(constraints: LandmarkConstraints, src_extent, dup_tol,
               contradiction_tol, fold_grid) -> str:
    payload = {
        "version": SOLVE_VERSION,
        "landmarks": constraints.fingerprint(),
        "src_extent": None if src_extent is None else [float(v) for v in src_extent],
        "dup_tol": float(dup_tol),
        "contradiction_tol": float(contradiction_tol),
        "fold_grid": int(fold_grid),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def solve_registration(constraints: LandmarkConstraints, *, src_extent=None,
                       dup_tol: float = 1.0, contradiction_tol: float = 10.0,
                       fold_grid: int = 120):
    """求解 TPS 并返回带版本的独立结果对象。

    成功返回 RegistrationSolve(即使诊断里含矛盾对应等非致命问题),
    无法建立模型时返回 RegistrationFailure —— 不用异常或默认模型掩盖失败。
    """
    if not isinstance(constraints, LandmarkConstraints):
        raise TypeError("solve_registration 需要 LandmarkConstraints")
    diag = run_all(constraints.src, constraints.dst, constraints.lam,
                   src_extent=src_extent, dup_tol=dup_tol,
                   contradiction_tol=contradiction_tol, fold_grid=fold_grid)
    public = {k: v for k, v in diag.items() if k != "model"}
    rid = _result_id(constraints, src_extent, dup_tol, contradiction_tol, fold_grid)
    errors = tuple(diag.get("errors", ()))
    warnings = tuple(diag.get("warnings", ()))
    if "model" not in diag:
        return RegistrationFailure(
            version=SOLVE_VERSION, result_id=rid, constraints=constraints,
            diagnostics=public, errors=errors, warnings=warnings)
    return RegistrationSolve(
        version=SOLVE_VERSION, result_id=rid, constraints=constraints,
        model=diag["model"], diagnostics=public, errors=errors, warnings=warnings)

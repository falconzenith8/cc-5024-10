"""古地图影像配准核心包: 地标约束、求解结果、变形场、诊断、不确定性、图像反采样。

分层:
  - kernel:   地标约束 -> 版本化求解结果(成功/失败为明确返回类型);
  - field:    变形求值(有版本的变形场);
  - resample: 图像反采样(有版本、显式容差);
  - tps / diagnostics / uncertainty / export / synth: 数值与辅助工具;
  - warp:     旧入口的兼容适配(古图配准既有调用方)。
"""
from .tps import (TPSModel, affine_part, bending_energy, evaluate, fit_tps,
                  jacobian, kernel)
from .diagnostics import run_all as run_diagnostics
from .uncertainty import prediction_variance, sigma_hat_from_loocv, uncertainty_grid
from .kernel import (LandmarkConstraintError, LandmarkConstraints,
                     RegistrationError, RegistrationFailure,
                     RegistrationNotSolved, RegistrationSolve, solve_registration)
from .field import (DeformationField, DisplacementFrame, FieldError,
                    field_from_model, legacy_field_from_model)
from .resample import (ResampleError, ResampleResult, invert_map, resample_field,
                       sample_bilinear)
from .warp import resample_to_grid
from .export import model_from_dict, model_to_dict

__all__ = [
    "TPSModel", "fit_tps", "evaluate", "jacobian", "kernel", "bending_energy",
    "affine_part", "run_diagnostics", "prediction_variance",
    "sigma_hat_from_loocv", "uncertainty_grid", "invert_map",
    "resample_to_grid", "sample_bilinear", "model_to_dict", "model_from_dict",
    "LandmarkConstraints", "LandmarkConstraintError", "solve_registration",
    "RegistrationSolve", "RegistrationFailure", "RegistrationError",
    "RegistrationNotSolved", "DeformationField", "DisplacementFrame",
    "FieldError", "field_from_model", "legacy_field_from_model",
    "ResampleResult", "ResampleError", "resample_field",
]

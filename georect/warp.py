"""兼容适配层: 保留旧入口 (invert_map / sample_bilinear / resample_to_grid)。

求逆与双线性采样的数值实现已迁至 georect.resample, 变形场版本对象在
georect.field。这里只做适配, 不复制公式, 保证旧调用方(古图配准及
外部脚本)看到的行为与结果逐字节不变:

  - invert_map / sample_bilinear 直接转发到 resample 层;
  - resample_to_grid(model, ...) 接受裸 TPSModel, 用内容指纹显式标注
    变形场来源, 并返回旧 dict 形状(未传 tol 时仍按输出像素 0.01 推导,
    与既有行为一致)。

新代码应使用 kernel.solve_registration + field.field() +
resample.resample_field, 以拿到带版本、来源明确的独立结果对象。
"""
from __future__ import annotations

import numpy as np

from .field import legacy_field_from_model
from .resample import (FAIL_NAMES, FAIL_NO_CONVERGE, FAIL_NONFINITE,
                       FAIL_OK, FAIL_SINGULAR, ResampleResult, ResampleError,
                       invert_map, resample_field, sample_bilinear)
from .tps import TPSModel

__all__ = [
    "FAIL_OK", "FAIL_SINGULAR", "FAIL_NO_CONVERGE", "FAIL_NONFINITE",
    "FAIL_NAMES", "invert_map", "sample_bilinear", "resample_to_grid",
    "ResampleResult", "ResampleError",
]


def resample_to_grid(model: TPSModel, img: np.ndarray, out_extent,
                     out_w: int, out_h: int, tol: float | None = None,
                     max_iter: int = 30) -> dict:
    """旧入口: 裸模型 + 隐式容差 -> 旧 dict 形状。

    tol=None 时按既有政策取最小输出像素的 0.01; 其余参数直接透传。
    """
    if tol is None:
        xmin, ymin, xmax, ymax = (float(v) for v in out_extent)
        px = (xmax - xmin) / int(out_w)
        py = (ymax - ymin) / int(out_h)
        tol = 0.01 * min(px, py)
    field = legacy_field_from_model(model)
    out = resample_field(field, img, out_extent, int(out_w), int(out_h),
                         tol=tol, max_iter=max_iter)
    return out.to_legacy_dict()

"""合成数据工具: 用已知变形场生成"扫描图", 供测试与演示。

scanned(x) = reference(f_true(x)) —— 对源平面上每个像素 x, 在参考图的
f_true(x) 处采样。这样 f_true 就是我们试图从地标恢复的真值变形。
"""
from __future__ import annotations

import numpy as np

from .warp import sample_bilinear


def forward_sample(field_fn, img: np.ndarray, out_w: int, out_h: int):
    """用已知变形场 field_fn(源->目标像素坐标) 从 img 生成扫描图。

    返回 (out_h, out_w, C) float 数组; 目标图外的位置为 0。
    """
    xs = np.arange(out_w, dtype=np.float64) + 0.5
    ys = np.arange(out_h, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    tgt = field_fn(pts)
    vals, valid = sample_bilinear(img, tgt[:, 0], tgt[:, 1])
    vals[~valid] = 0.0
    return vals.reshape(out_h, out_w, img.shape[2] if img.ndim == 3 else 1)


def smooth_field(size, bumps, base=None):
    """构造平滑变形场: f(x) = base(x) + Σ A_k·v_k·exp(-|x-c_k|²/2σ_k²)。

    size = (W, H) 源平面尺寸; bumps = [(cx, cy, sigma, vx, vy), ...];
    base = (c0, C1) 仿射项, 缺省为恒等。
    """
    W, H = size
    if base is None:
        c0 = np.zeros(2)
        C1 = np.eye(2)
    else:
        c0, C1 = base

    def field(pts):
        pts = np.asarray(pts, dtype=np.float64)
        out = c0 + pts @ C1
        for cx, cy, sigma, vx, vy in bumps:
            d2 = ((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) / (2 * sigma * sigma)
            g = np.exp(-d2)
            out = out + g[:, None] * np.array([vx, vy])
        return out

    return field

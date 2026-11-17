"""未约束区域的不确定性估计。

TPS 等价于以 U(r)=r²·ln r 为广义协方差的内在随机场克里金 (IRF-k kriging)。
预测方差(相对量)由系统矩阵给出:

    φ(x) = m(x)ᵀ M⁻¹ m(x),   m(x) = [U(|x-p_1|),...,U(|x-p_n|), 1, x, y]ᵀ
    Var[f(x)] = -σ² · φ(x)          (TPS 核为条件负定, 故取负号)

性质: 控制点处 φ=0(λ=0 时), 离控制点越远越大 —— 直观表达
"离地标越远越没把握"。尺度 σ² 用留一残差均方估计, 使不确定性
以目标平面坐标单位(如米/度)给出, 而不是无量纲相对值。

凸包以外的区域属于外推, 单独标记为"未约束/外推", 与拟合误差分开呈现。
"""
from __future__ import annotations

import numpy as np

from .diagnostics import convex_hull, loocv_residuals
from .tps import TPSModel, kernel, _dist_matrix


def sigma_hat_from_loocv(src: np.ndarray, dst: np.ndarray, lam: float) -> float:
    """以留一残差均方估计预测误差尺度 σ (每坐标分量)。"""
    res = loocv_residuals(np.asarray(src, float), np.asarray(dst, float), lam)
    res = res[np.isfinite(res).all(1)]
    if len(res) == 0:
        return float("nan")
    return float(np.sqrt((res ** 2).sum() / (2 * len(res))))


def prediction_variance(model: TPSModel, X) -> np.ndarray:
    """各点的相对预测方差 -φ(x) (>=0, 控制点处为 0)。

    在归一化坐标中求解以保证数值稳定; φ 是坐标变换下的不变量。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    from .tps import normalized_system
    s_norm = model.normalized_src()
    M = normalized_system(s_norm, model.normalized_lambda())
    Xn = (X - model.center) / model.scale
    U = kernel(_dist_matrix(Xn, s_norm))  # (m,n)
    m_all = np.concatenate([U, np.ones((len(X), 1)), Xn], axis=1)  # (m, n+3)
    sol = np.linalg.solve(M, m_all.T)  # (n+3, m)
    phi = np.einsum("im,im->m", m_all.T, sol)
    var = -phi
    var[var < 0] = 0.0  # 数值误差截断
    return var


def in_convex_polygon(pts: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """判断点是否在凸多边形(逆时针顶点)内。"""
    pts = np.asarray(pts, dtype=np.float64)
    if len(hull) < 3:
        return np.zeros(len(pts), dtype=bool)
    inside = np.ones(len(pts), dtype=bool)
    h = np.asarray(hull, dtype=np.float64)
    for k in range(len(h)):
        a = h[k]
        b = h[(k + 1) % len(h)]
        cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
        inside &= cross >= -1e-12
    return inside


def uncertainty_grid(model: TPSModel, dst: np.ndarray, sigma: float,
                     extent, nx: int = 90, ny: int = 90):
    """在目标平面网格上计算不确定性标准差与凸包掩膜。

    extent = (xmin, ymin, xmax, ymax) 目标平面范围。
    返回 dict: xs, ys, std (ny,nx), inside_hull (ny,nx) bool, hull 顶点。
    """
    xmin, ymin, xmax, ymax = extent
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    var = prediction_variance(model, pts)
    std = sigma * np.sqrt(var)
    hull = convex_hull(np.asarray(dst, dtype=np.float64))
    inside = in_convex_polygon(pts, hull)
    return {
        "xs": xs,
        "ys": ys,
        "std": std.reshape(ny, nx),
        "inside_hull": inside.reshape(ny, nx),
        "hull": hull,
        "sigma": sigma,
    }

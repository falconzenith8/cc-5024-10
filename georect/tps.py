"""二维薄板样条 (Thin Plate Spline, TPS) 平滑非刚性变形模型。

模型形式（显式、可解释）:
    f(x) = c0 + C1·x + Σ_i w_i · U(|x - p_i|),   U(r) = r²·ln r
其中 p_i 为源平面控制点，q_i 为目标平面对应点。
  - 仿射项 (c0, C1) 表达整体平移/旋转/缩放/错切；
  - 核权重 w_i 表达局部非刚性修正，随距离平滑衰减。

拟合目标（可解释正则化）:
    min  Σ_i |f(p_i) - q_i|² + λ · E_bend,   E_bend = tr(Wᵀ K W)
λ（弯曲能量惩罚）含义明确:
  λ = 0   精确插值，曲面可高度弯曲；
  λ → ∞   退化为纯仿射变换（最"刚"的平滑解）。
弯曲能量 E_bend 以源平面用户单位计，可直接读出衡量变形非刚性程度。

数值实现（对结果无影响，仅为稳定性）:
  线性系统在归一化坐标 s = (x - center)/scale 中求解, 核取无量纲
  Ũ(ρ) = ρ²·ln ρ。数学上可以证明: 在约束 PᵀW=0 下, 该归一化与
  用户坐标系直接求解给出同一个样条(仅常数项被吸收进 c0),
  但条件数不再随图像尺寸爆炸。求得的权重会变换回用户坐标系,
  导出的模型 (src, W, C) 与 λ 均在用户坐标下解释。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KERNEL_NAME = "tps-r2logr"


def kernel(r: np.ndarray) -> np.ndarray:
    """TPS 径向基 U(r) = r²·ln r, U(0) = 0 (用户坐标)。"""
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    m = r > 0
    rm = r[m]
    out[m] = rm * rm * np.log(rm)
    return out


def kernel_grad_factor(r: np.ndarray) -> np.ndarray:
    """U'(r)/r = 1 + 2·ln r (r>0); r=0 处极限为 0 (r·ln r → 0)。"""
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    m = r > 0
    out[m] = 1.0 + 2.0 * np.log(r[m])
    return out


@dataclass
class TPSModel:
    """拟合得到的变形模型, 全部量均在用户坐标系下解释。

    center/scale 记录求解时使用的归一化(仅不确定性/条件数等
    数值敏感计算会用到); 用 W/C 直接求值不需要它们。
    """

    src: np.ndarray  # (n,2) 源平面控制点(用户坐标)
    W: np.ndarray  # (n,2) 非刚性核权重(用户坐标核 U=r²ln r)
    C: np.ndarray  # (3,2) 仿射系数, 行对应 [1, x, y]
    lam: float  # 用户坐标下的正则化强度
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 1.0

    @property
    def n(self) -> int:
        return int(self.src.shape[0])

    def normalized_src(self) -> np.ndarray:
        return (self.src - self.center) / self.scale

    def normalized_lambda(self) -> float:
        """归一化坐标中的等效 λ (弯曲能量随尺度平方变化)。"""
        return self.lam / (self.scale ** 2)


def _dist_matrix(A: np.ndarray, B: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """|A_i - B_j| 距离矩阵, 分块避免大中间数组。"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] <= chunk:
        d = A[:, None, :] - B[None, :, :]
        return np.sqrt((d * d).sum(-1))
    parts = []
    for s in range(0, A.shape[0], chunk):
        d = A[s : s + chunk, None, :] - B[None, :, :]
        parts.append(np.sqrt((d * d).sum(-1)))
    return np.vstack(parts)


def system_matrix(src: np.ndarray, lam: float) -> np.ndarray:
    """用户坐标 TPS 系统矩阵 M = [[K + λI, P], [Pᵀ, 0]] (仅用于小系统/检查)。"""
    n = src.shape[0]
    K = kernel(_dist_matrix(src, src))
    P = np.column_stack([np.ones(n), src])
    M = np.zeros((n + 3, n + 3))
    M[:n, :n] = K + lam * np.eye(n)
    M[:n, n:] = P
    M[n:, :n] = P.T
    return M


def normalized_system(src_norm: np.ndarray, lam_norm: float) -> np.ndarray:
    """归一化坐标的系统矩阵(无量纲核 ρ²lnρ), 条件数与图像尺寸无关。"""
    n = src_norm.shape[0]
    K = kernel(_dist_matrix(src_norm, src_norm))
    P = np.column_stack([np.ones(n), src_norm])
    M = np.zeros((n + 3, n + 3))
    M[:n, :n] = K + lam_norm * np.eye(n)
    M[:n, n:] = P
    M[n:, :n] = P.T
    return M


def fit_tps(src, dst, lam: float = 0.0) -> TPSModel:
    """求解 TPS。输入非法时抛 ValueError; 系统奇异时抛 LinAlgError。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src 应为 (n,2) 数组")
    if dst.shape != src.shape:
        raise ValueError("dst 形状应与 src 一致 (n,2)")
    if src.shape[0] < 3:
        raise ValueError("至少需要 3 个控制点才能拟合 TPS")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("控制点坐标包含 NaN 或 Inf, 拒绝拟合")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("λ 必须为有限非负数")

    n = src.shape[0]
    center = src.mean(0)
    scale = float(np.sqrt(((src - center) ** 2).sum(1).mean()))
    if scale < 1e-12:
        raise np.linalg.LinAlgError("控制点全部重合, 无法拟合")
    s = (src - center) / scale
    lam_norm = lam / (scale ** 2)

    M = normalized_system(s, lam_norm)
    rhs = np.vstack([dst, np.zeros((3, 2))])
    sol = np.linalg.solve(M, rhs)  # 枢轴精确为零时抛 LinAlgError
    # 显式退化检查: 秩缺系统的枢轴未必精确为零, 验证解的残差
    res = M @ sol - rhs
    rhs_scale = max(1.0, float(np.abs(rhs).max()))
    if not np.isfinite(sol).all() or float(np.abs(res).max()) > 1e-8 * rhs_scale:
        raise np.linalg.LinAlgError("TPS 系统奇异或严重病态(通常由重复/共线点引起)")
    V, A = sol[:n], sol[n:]

    # 变换回用户坐标(推导见模块 docstring):
    # f(x) = a0 + A1·(x-c)/r + Σ v_i Ũ(|x-p_i|/r)
    #      = c0' + C1'·x + Σ w_i' U(|x-p_i|) + 常数(吸收进 c0')
    W = V / scale ** 2
    C1 = A[1:] / scale
    c0 = A[0] - center @ C1 - (np.log(scale) / scale ** 2) * (V * (src ** 2).sum(1)[:, None]).sum(0)
    C = np.vstack([c0, C1])
    return TPSModel(src=src, W=W, C=C, lam=float(lam), center=center, scale=scale)


def evaluate(model: TPSModel, X, chunk: int = 20000) -> np.ndarray:
    """在点集 X (m,2) 上求变形 f(X) -> (m,2)。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    U = kernel(_dist_matrix(X, model.src, chunk))
    return model.C[0] + X @ model.C[1:] + U @ model.W


def jacobian(model: TPSModel, X, chunk: int = 20000) -> np.ndarray:
    """变形雅可比 J[m][a,b] = ∂f_a/∂x_b, 返回 (m,2,2)。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    m = X.shape[0]
    J = np.empty((m, 2, 2))
    for s in range(0, m, chunk):
        Xc = X[s : s + chunk]
        D = Xc[:, None, :] - model.src[None, :, :]  # (mc,n,2)
        r = np.sqrt((D * D).sum(-1))  # (mc,n)
        g = kernel_grad_factor(r)  # (mc,n)
        # J = C1ᵀ + Σ_i g_i · w_i ⊗ (x - p_i)   (J[a,b] = ∂f_a/∂x_b)
        Jc = np.broadcast_to(model.C[1:].T, (Xc.shape[0], 2, 2)).copy()
        Jc += np.einsum("mi,ia,mib->mab", g, model.W, D)
        J[s : s + chunk] = Jc
    return J


def bending_energy(model: TPSModel) -> float:
    """弯曲能量 tr(Wᵀ K W)(用户坐标)—— 被 λ 直接惩罚的量, 越大越"皱"。"""
    K = kernel(_dist_matrix(model.src, model.src))
    return float(np.einsum("ia,ij,ja->", model.W, K, model.W))


def affine_part(model: TPSModel):
    """返回 (c0, C1): 仿射平移向量与 2x2 线性部分。"""
    return model.C[0].copy(), model.C[1:].copy()

import numpy as np
import pytest

from georect.tps import (affine_part, bending_energy, evaluate, fit_tps,
                         jacobian)


def rng_pts(n, seed=0, scale=500.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, scale, (n, 2))


def test_exact_interpolation():
    src = rng_pts(10, seed=1)
    dst = rng_pts(10, seed=2)
    m = fit_tps(src, dst, lam=0.0)
    got = evaluate(m, src)
    assert np.allclose(got, dst, atol=1e-6)


def test_affine_reproduced_exactly():
    src = rng_pts(8, seed=3)
    c0 = np.array([13.0, -7.0])
    C1 = np.array([[1.2, 0.1], [-0.05, 0.9]])
    dst = c0 + src @ C1
    for lam in (0.0, 10.0):
        m = fit_tps(src, dst, lam=lam)
        assert np.allclose(evaluate(m, src), dst, atol=1e-6)
        assert bending_energy(m) < 1e-12  # 纯仿射无弯曲
        got_c0, got_C1 = affine_part(m)
        assert np.allclose(got_c0, c0, atol=1e-6)
        assert np.allclose(got_C1, C1, atol=1e-6)


def test_jacobian_matches_finite_difference():
    src = rng_pts(9, seed=4)
    dst = rng_pts(9, seed=5)
    m = fit_tps(src, dst, lam=1.0)
    X = rng_pts(20, seed=6)
    J = jacobian(m, X)
    eps = 1e-5
    for d in range(2):
        dp = np.zeros(2)
        dp[d] = eps
        fp = evaluate(m, X + dp)
        fm = evaluate(m, X - dp)
        fd = (fp - fm) / (2 * eps)  # (m,2) = ∂f/∂x_d
        assert np.allclose(J[:, :, d], fd, rtol=1e-4, atol=1e-4)


def test_smoothing_trades_fit_for_stiffness():
    rng = np.random.default_rng(7)
    src = rng_pts(12, seed=7)
    clean = src @ np.array([[1.0, 0.2], [0.0, 1.0]])
    noisy = clean + rng.normal(0, 20.0, clean.shape)
    m0 = fit_tps(src, noisy, lam=0.0)
    m1 = fit_tps(src, noisy, lam=500.0)
    r0 = np.linalg.norm(evaluate(m0, src) - noisy, axis=1).mean()
    r1 = np.linalg.norm(evaluate(m1, src) - noisy, axis=1).mean()
    assert r0 < 1e-6 < r1  # λ=0 精确插值, λ>0 允许残差
    assert bending_energy(m1) < bending_energy(m0)  # 更刚硬


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        fit_tps([[0, 0], [1, 1]], [[0, 0], [1, 1]])  # 点数不足
    with pytest.raises(ValueError):
        fit_tps([[0, 0], [1, 0], [np.nan, 1]], [[0, 0], [1, 0], [0, 1]])
    with pytest.raises(ValueError):
        fit_tps([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]], lam=-1)
    with pytest.raises(np.linalg.LinAlgError):
        # 完全重复点 -> 系统奇异
        fit_tps([[0, 0], [1, 0], [0, 1], [0, 0]], [[0, 0], [1, 0], [0, 1], [5, 5]])

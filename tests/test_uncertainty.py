import numpy as np

from georect.synth import smooth_field
from georect.tps import fit_tps
from georect.uncertainty import (in_convex_polygon, prediction_variance,
                                 sigma_hat_from_loocv, uncertainty_grid)
from georect.diagnostics import convex_hull


def make_model(seed=0):
    rng = np.random.default_rng(seed)
    src = rng.uniform(0, 500, (10, 2))
    field = smooth_field((500, 500), [(250, 250, 150, 20, -15)])
    dst = field(src)
    return fit_tps(src, dst, lam=0.0), src, dst


def test_variance_zero_at_controls_positive_away():
    m, src, _ = make_model()
    var_at = prediction_variance(m, src)
    assert np.all(var_at < 1e-6)
    far = np.array([[250.0, 250.0], [-300.0, -300.0], [1000.0, 1000.0]])
    var_far = prediction_variance(m, far)
    assert (var_far > 0).all()
    # 单调趋势: 凸包内中心的不确定性应小于远处外推点
    assert var_far[0] < var_far[1]


def test_hull_membership():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    hull = convex_hull(pts)
    inside = in_convex_polygon(np.array([[5, 5], [15, 5], [-1, 0]], float), hull)
    assert inside.tolist() == [True, False, False]


def test_uncertainty_grid_marks_extrapolation():
    m, src, dst = make_model()
    sigma = sigma_hat_from_loocv(src, dst, lam=0.0)
    assert np.isfinite(sigma) and sigma > 0
    g = uncertainty_grid(m, dst, sigma, (-200, -200, 700, 700), nx=40, ny=40)
    assert g["std"].shape == (40, 40)
    assert g["inside_hull"].any()
    assert (~g["inside_hull"]).any()  # 网格外延有外推区
    # 凸包外角点的 std 应大于控制点附近
    corner_std = g["std"][0, 0]
    i0 = np.argmin(np.abs(g["xs"] - src[0, 0]))
    j0 = np.argmin(np.abs(g["ys"] - src[0, 1]))
    near_std = g["std"][j0, i0]
    assert corner_std > near_std

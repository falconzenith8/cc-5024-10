import numpy as np

from georect.diagnostics import (find_contradictions, find_duplicates,
                                 flag_outliers, fold_analysis,
                                 geometry_assessment, loocv_residuals, run_all)
from georect.synth import smooth_field
from georect.tps import fit_tps


def grid_pts(nx, ny, w=1000.0, h=800.0):
    xs = np.linspace(0.1 * w, 0.9 * w, nx)
    ys = np.linspace(0.1 * h, 0.9 * h, ny)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def test_duplicate_and_contradiction_detection():
    src = np.array([[0, 0], [100, 0], [0, 100], [0.2, 0.1]], float)
    dst = np.array([[0, 0], [100, 0], [0, 100], [50, 50]], float)
    dups = find_duplicates(src, tol=1.0)
    assert (0, 3, dups[0][2]) in [(i, j, d) for i, j, d in dups]
    contra = find_contradictions(src, dst, src_tol=1.0, dst_tol=10.0)
    assert len(contra) == 1 and {contra[0][0], contra[0][1]} == {0, 3}


def test_collinear_geometry_flagged():
    t = np.linspace(0, 500, 6)
    line = np.column_stack([t, 2 * t + 1])
    geo = geometry_assessment(line)
    assert geo["near_collinear"]
    good = geometry_assessment(grid_pts(3, 3))
    assert not good["near_collinear"]


def test_loocv_flags_wrong_landmark():
    src = grid_pts(4, 4)
    field = smooth_field((1000, 800), [(500, 400, 200, 30, -20)])
    dst = field(src)
    dst_bad = dst.copy()
    dst_bad[5] += np.array([120.0, -80.0])  # 注入错误地标
    loocv = loocv_residuals(src, dst_bad, lam=0.0)
    flags = flag_outliers(loocv)
    norms = np.linalg.norm(loocv, axis=1)
    assert flags[5]
    assert norms[5] == norms.max()
    assert flags.sum() <= 2  # 错误地标及其被波及的最近邻可能被一并标出


def test_loocv_clean_when_data_consistent():
    src = grid_pts(4, 4)
    field = smooth_field((1000, 800), [(300, 300, 250, 40, 10)])
    dst = field(src)
    loocv = loocv_residuals(src, dst, lam=0.0)
    assert not flag_outliers(loocv).any()
    # 留一预测误差应远小于变形幅度(40px)的量级
    assert np.linalg.norm(loocv, axis=1).max() < 15.0


def test_fold_detected_for_contradictory_targets():
    src = grid_pts(5, 5)
    dst = src.copy()
    # 交换两个相邻目标点 -> 局部翻折
    i, j = 11, 12
    dst[[i, j]] = dst[[j, i]]
    m = fit_tps(src, dst, lam=0.0)
    fa = fold_analysis(m, (0, 0, 1000, 800), n_grid=100)
    assert fa["finite"]
    assert fa["n_fold"] > 0


def test_no_fold_for_smooth_field():
    src = grid_pts(5, 5)
    field = smooth_field((1000, 800), [(500, 400, 300, 60, 30)])
    dst = field(src)
    m = fit_tps(src, dst, lam=0.0)
    fa = fold_analysis(m, (0, 0, 1000, 800), n_grid=100)
    assert fa["n_fold"] == 0


def test_run_all_reports_errors_gracefully():
    # 完全重复点 -> 系统奇异, 应记录错误而非抛异常
    src = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], float)
    dst = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], float)
    out = run_all(src, dst, lam=0.0, dup_tol=1e-6)
    assert out["errors"]
    assert "model" not in out

    # NaN 输入
    out2 = run_all([[0, 0], [1, 0], [np.nan, 1]], [[0, 0], [1, 0], [0, 1]], 0.0)
    assert out2["errors"]


def test_run_all_smooth_case_clean():
    src = grid_pts(4, 4)
    field = smooth_field((1000, 800), [(500, 400, 300, 50, 20)])
    dst = field(src)
    out = run_all(src, dst, lam=0.0, src_extent=(0, 0, 1000, 800))
    assert not out["errors"]
    assert out["folds"]["n_fold"] == 0
    assert out["max_error"] < 1e-6


def test_loocv_undefined_with_too_few_points():
    """3 点是合法的拟合规模, 但留一后不足 3 点: LOOCV 必须置 NaN 而非抛异常。"""
    src = np.array([[0, 0], [10, 0], [0, 10]], float)
    dst = np.array([[5, 5], [15, 0], [0, 10]], float)
    loocv = loocv_residuals(src, dst, 0.0)
    assert loocv.shape == (3, 2)
    assert np.isnan(loocv).all()
    out = run_all(src, dst, 0.0, src_extent=(0, 0, 10, 10))
    assert not out["errors"]
    assert out["model"].n == 3
    assert np.isnan(out["loocv_rms"])

import numpy as np

from georect.diagnostics import fold_analysis
from georect.synth import forward_sample, smooth_field
from georect.tps import evaluate, fit_tps
from georect.warp import invert_map, resample_to_grid

W, H = 1000, 800


def grid_pts(nx, ny, mx=0.08):
    xs = np.linspace(mx * W, (1 - mx) * W, nx)
    ys = np.linspace(mx * H, (1 - mx) * H, ny)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def make_ref_image(w=W, h=H):
    """合成"参考地图": 平滑周期图案, 对配准误差敏感且可数值比较。"""
    xs = np.arange(w)[None, :].astype(float)
    ys = np.arange(h)[:, None].astype(float)
    r = 127 + 120 * np.sin(2 * np.pi * xs / 160) * np.ones((h, 1))
    g = 127 + 120 * np.cos(2 * np.pi * ys / 130) * np.ones((1, w))
    b = 127 + 120 * np.sin(2 * np.pi * (xs + ys) / 220) * np.ones((h, w))
    return np.dstack([r, g, b])


def test_known_smooth_deformation_recovered():
    field = smooth_field((W, H), [(300, 300, 220, 45, -25), (750, 600, 180, -30, 35)])
    src = grid_pts(6, 5)
    dst = field(src)
    m = fit_tps(src, dst, lam=0.0)
    test = grid_pts(20, 16, mx=0.1)
    err = np.linalg.norm(evaluate(m, test) - field(test), axis=1)
    assert err.max() < 3.0  # 坐标残差(像素), 均值远小于 1px
    assert err.mean() < 0.6
    fa = fold_analysis(m, (0, 0, W, H), n_grid=80)
    assert fa["n_fold"] == 0  # 无网格翻折


def test_local_paper_stretch_recovered_with_local_landmarks():
    # 局部纸张伸缩: 紧凑强拉伸
    stretch = (620.0, 430.0, 55.0, 55.0, -40.0)
    field = smooth_field((W, H), [stretch])
    src_sparse = grid_pts(6, 5)
    test = grid_pts(20, 16, mx=0.1)

    m_sparse = fit_tps(src_sparse, field(src_sparse), lam=0.0)
    err_sparse = np.linalg.norm(evaluate(m_sparse, test) - field(test), axis=1)

    # 在伸缩区附近加密地标(模拟用户在地物密集处加点)
    rng = np.random.default_rng(0)
    local = np.column_stack([
        rng.uniform(stretch[0] - 120, stretch[0] + 120, 12),
        rng.uniform(stretch[1] - 120, stretch[1] + 120, 12),
    ])
    src = np.vstack([src_sparse, local])
    m = fit_tps(src, field(src), lam=0.0)
    err = np.linalg.norm(evaluate(m, test) - field(test), axis=1)

    # 加密地标后残差显著下降, 且整体均值小
    assert err.max() < 0.7 * err_sparse.max()
    assert err.mean() < 1.5


def test_resample_matches_reference_image():
    field = smooth_field((W, H), [(400, 350, 250, 40, 20)])
    ref = make_ref_image()
    scanned = forward_sample(field, ref, W, H)  # scanned(x)=ref(field(x))
    src = grid_pts(6, 5)
    dst = field(src)
    m = fit_tps(src, dst, lam=0.0)
    # 输出范围平移半像素, 使输出像素中心与 ref[::2,::2] 的像素中心对齐
    out = resample_to_grid(m, scanned, (-0.5, -0.5, W - 0.5, H - 0.5), W // 2, H // 2)
    img = out["rgba"]
    assert out["stats"]["n_failed"] == 0
    # 输出第 0 行对应 ymax(世界坐标 y 向上), 翻转后与参考图数组对齐
    aligned = img[::-1]
    ref_small = ref[::2, ::2][: H // 2, : W // 2]
    valid = out["valid"][::-1]
    assert valid.mean() > 0.95
    # 只比较内部区域: 扫描图边界带在正演时落在参考图外被置 0, 属预期无数据
    mg = 40
    inner = np.zeros_like(valid)
    inner[mg:-mg, mg:-mg] = True
    sel = valid & inner
    diff = np.abs(aligned[:, :, :3].astype(float) - ref_small.astype(float))
    assert diff[sel].mean() < 3.0  # 平均像素差(0-255)


def test_no_content_fabricated_beyond_boundary():
    field = smooth_field((W, H), [(500, 400, 300, 30, 30)])
    ref = make_ref_image()
    scanned = forward_sample(field, ref, W, H)
    src = grid_pts(5, 4)
    dst = field(src)
    m = fit_tps(src, dst, lam=0.0)
    # 输出网格明显超出源图覆盖范围
    out = resample_to_grid(m, scanned, (-400, -400, W + 400, H + 400), 350, 350)
    assert out["stats"]["n_outside"] > 0
    ov = out["outside"]
    # 越界像素必须是透明无数据, 而不是编造的内容
    assert (out["rgba"][ov, 3] == 0).all()
    # 角部区域应当几乎全部无数据
    assert ov[:20, :20].mean() > 0.9


def test_inversion_reports_failures_in_folded_map():
    src = grid_pts(5, 5)
    dst = src.copy()
    dst[[11, 12]] = dst[[12, 11]]  # 制造折叠
    m = fit_tps(src, dst, lam=0.0)
    img = make_ref_image()
    out = resample_to_grid(m, img, (0, 0, W, H), 200, 160)
    s = out["stats"]
    assert s["n_failed"] > 0  # 折叠区求逆失败被显式报告
    assert np.isfinite(out["rgba"]).all()
    # 失败像素 alpha=0
    assert (out["rgba"][out["failed"], 3] == 0).all()


def test_invert_map_roundtrip():
    field = smooth_field((W, H), [(500, 400, 260, 35, -20)])
    src = grid_pts(6, 5)
    dst = field(src)
    m = fit_tps(src, dst, lam=0.0)
    Y = grid_pts(15, 12, mx=0.15)
    X, info = invert_map(m, field(Y), tol=1e-6)
    assert info["converged"].all()
    # 求逆精度受拟合残差限制, 应在亚像素量级
    assert np.linalg.norm(X - Y, axis=1).max() < 1.0

"""控制点与变形场的诊断: 退化检查、矛盾对应、折叠检测。

原则: 不能只看"几个点拟合上了"就宣称整幅图精确配准。
这里集中提供可解释的检查项, 所有结果都会展示给用户:

- 重复点 / 矛盾对应: 源点几乎重合但目标不同 -> 数据错误;
- 近共线 / 几何退化: 控制点近似落在一条直线上, 仿射方向不可辨识;
- 系统条件数: 过大说明几何布局接近奇异, 结果对噪声敏感;
- 留一交叉验证 (LOOCV): 逐点留出重拟合, 残差显著大者疑似错误地标;
- 折叠检测: 在源域网格上检查雅可比行列式符号, det<=0 即发生折叠
  (变形场在该处把平面"翻折", 输出图不可逆, 必须警告);
- 非有限检查: 变形场任何位置出现 NaN/Inf 都必须报告。
"""
from __future__ import annotations

import numpy as np

from .tps import TPSModel, evaluate, fit_tps, jacobian, system_matrix


def find_duplicates(pts: np.ndarray, tol: float):
    """距离小于 tol 的点对, 返回 [(i, j, dist), ...]。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    out = []
    for i in range(n):
        d = np.sqrt(((pts[i + 1 :] - pts[i]) ** 2).sum(-1))
        for k in np.nonzero(d < tol)[0]:
            out.append((i, i + 1 + int(k), float(d[k])))
    return out


def find_contradictions(src: np.ndarray, dst: np.ndarray, src_tol: float, dst_tol: float):
    """源点几乎重合、目标点却明显不同的对应 -> 矛盾对应。"""
    out = []
    for i, j, d in find_duplicates(src, src_tol):
        dd = float(np.linalg.norm(dst[i] - dst[j]))
        if dd > dst_tol:
            out.append((i, j, d, dd))
    return out


def geometry_assessment(src: np.ndarray, collinear_ratio: float = 1e-3):
    """控制点几何布局评估。

    返回奇异值比(越小越接近共线)、是否近共线、凸包面积、外接框对角线。
    """
    src = np.asarray(src, dtype=np.float64)
    n = len(src)
    c = src - src.mean(0)
    svals = np.linalg.svd(c, compute_uv=False) if n >= 2 else np.array([0.0, 0.0])
    s0 = float(svals[0]) if len(svals) > 0 else 0.0
    s1 = float(svals[1]) if len(svals) > 1 else 0.0
    ratio = (s1 / s0) if s0 > 0 else 0.0
    diag = float(np.linalg.norm(src.max(0) - src.min(0))) if n >= 2 else 0.0
    hull = convex_hull(src)
    area = _polygon_area(hull) if len(hull) >= 3 else 0.0
    return {
        "n": n,
        "singular_ratio": ratio,
        "near_collinear": bool(ratio < collinear_ratio),
        "hull_area": float(area),
        "bbox_diag": diag,
    }


def condition_number(src: np.ndarray, lam: float) -> float:
    """归一化坐标下 TPS 系统矩阵条件数(与图像尺寸无关); 奇异时返回 inf。"""
    from .tps import normalized_system
    src = np.asarray(src, dtype=np.float64)
    center = src.mean(0)
    scale = float(np.sqrt(((src - center) ** 2).sum(1).mean()))
    if scale < 1e-12:
        return float("inf")
    try:
        M = normalized_system((src - center) / scale, lam / scale ** 2)
        return float(np.linalg.cond(M))
    except np.linalg.LinAlgError:
        return float("inf")


def loocv_residuals(src: np.ndarray, dst: np.ndarray, lam: float) -> np.ndarray:
    """留一交叉验证: 留出第 i 点重拟合, 在 src[i] 处预测并与 dst[i] 比较。

    返回 (n,2) 残差; 若留出后点数不足或系统奇异, 该行置 NaN 并在汇总中说明。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    res = np.full((n, 2), np.nan)
    idx = np.arange(n)
    if n < 4:
        # 留出一点后不足 3 点, 无法重拟合; 明确置 NaN 而不是抛出未处理异常
        return res
    for i in range(n):
        keep = idx != i
        try:
            m = fit_tps(src[keep], dst[keep], lam)
            res[i] = evaluate(m, src[i : i + 1])[0] - dst[i]
        except (np.linalg.LinAlgError, ValueError):
            pass
    return res


def flag_outliers(loocv: np.ndarray, z: float = 6.0):
    """稳健标记疑似错误地标。返回 bool 数组。

    尺度用"下半部 MAD"(仅取不大于中位数的偏差的 MAD),
    避免多个异常值互相遮蔽(经典 MAD 在多异常时会膨胀失效);
    同时要求超过 3 倍中位数, 防止整体残差都很小时误报。
    """
    norms = np.linalg.norm(loocv, axis=1)
    finite = np.isfinite(norms)
    flags = np.zeros(len(loocv), dtype=bool)
    if finite.sum() < 4:
        return flags
    v = norms[finite]
    med = np.median(v)
    lower = v[v <= med]
    scale = 1.4826 * np.median(med - lower) if len(lower) else 0.0
    thresh = max(med + z * scale, 3.0 * med)
    if thresh <= 0:
        return flags
    flags[finite] = norms[finite] > thresh
    return flags


def fold_analysis(model: TPSModel, extent, n_grid: int = 120):
    """在源域网格上检查折叠: det J <= 0 (相对主符号) 即折叠。

    extent = (xmin, ymin, xmax, ymax), 通常取源图范围。
    主符号取 det 中位数的符号, 使检查对整体镜像(坐标系手性)不敏感。
    """
    xmin, ymin, xmax, ymax = extent
    xs = np.linspace(xmin, xmax, n_grid)
    ys = np.linspace(ymin, ymax, n_grid)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    J = jacobian(model, pts)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    finite = np.isfinite(det).all() and np.isfinite(pts).all()
    F = evaluate(model, pts)
    finite = bool(finite and np.isfinite(F).all())

    med = float(np.median(det))
    dominant = 1.0 if med >= 0 else -1.0
    signed = det * dominant
    fold_mask = signed <= 0
    cells = []
    if fold_mask.any():
        idx = np.nonzero(fold_mask)[0]
        cells = pts[idx].tolist()
    return {
        "finite": finite,
        "min_det": float(det.min()),
        "max_det": float(det.max()),
        "dominant_sign": dominant,
        "n_fold": int(fold_mask.sum()),
        "frac_fold": float(fold_mask.mean()),
        "fold_cells_src": cells,  # 源域坐标, 调用方可映射到目标域展示
    }


def convex_hull(points: np.ndarray) -> np.ndarray:
    """单调链凸包, 返回逆时针顶点 (h,2)。"""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1])


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def run_all(src, dst, lam: float, src_extent=None, dup_tol: float = 1.0,
            contradiction_tol: float = 10.0, fold_grid: int = 120):
    """汇总全部诊断。任何一步失败都记录到 errors, 不抛出。

    返回 dict, 含 model(成功时)、各项诊断与 errors 列表。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    out = {"errors": [], "warnings": []}

    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        out["errors"].append("控制点坐标包含 NaN/Inf")
        return out
    if len(src) < 3:
        out["errors"].append("控制点不足: 至少需要 3 个")
        return out

    out["duplicates_src"] = find_duplicates(src, dup_tol)
    out["duplicates_dst"] = find_duplicates(dst, dup_tol)
    out["contradictions"] = find_contradictions(src, dst, dup_tol, contradiction_tol)
    if out["duplicates_src"]:
        out["warnings"].append(f"源平面存在 {len(out['duplicates_src'])} 对重复点")
    if out["contradictions"]:
        out["errors"].append(
            f"发现 {len(out['contradictions'])} 对矛盾对应(源点重合但目标不同), 请修正地标")

    geo = geometry_assessment(src)
    out["geometry"] = geo
    if geo["near_collinear"]:
        out["warnings"].append("控制点近共线: 垂直于连线方向的变形不可辨识, 结果不可靠")

    cond = condition_number(src, lam)
    out["condition_number"] = cond
    if not np.isfinite(cond):
        out["errors"].append("TPS 系统矩阵奇异(通常由完全重复点引起)")
        return out
    if cond > 1e10:
        out["warnings"].append(f"系统条件数 {cond:.2e} 过大, 几何布局接近退化")

    try:
        model = fit_tps(src, dst, lam)
    except np.linalg.LinAlgError:
        out["errors"].append("TPS 系统求解失败(矩阵奇异), 请检查重复/共线点")
        return out
    out["model"] = model

    fit_res = evaluate(model, src) - dst
    out["fit_residuals"] = fit_res
    out["rms_error"] = float(np.sqrt((fit_res ** 2).sum(1).mean()))
    out["max_error"] = float(np.linalg.norm(fit_res, axis=1).max())

    loocv = loocv_residuals(src, dst, lam)
    out["loocv_residuals"] = loocv
    out["loocv_flags"] = flag_outliers(loocv)
    finite_loocv = loocv[np.isfinite(loocv).all(1)]
    out["loocv_rms"] = (
        float(np.sqrt((finite_loocv ** 2).sum(1).mean())) if len(finite_loocv) else float("nan")
    )
    if out["loocv_flags"].any():
        idx = np.nonzero(out["loocv_flags"])[0].tolist()
        out["warnings"].append(
            f"地标 {idx} 的留一残差显著偏大: 可能是错误对应, 也可能是局部变形过于复杂")

    if src_extent is not None:
        out["folds"] = fold_analysis(model, src_extent, n_grid=fold_grid)
        if not out["folds"]["finite"]:
            out["errors"].append("变形场出现非有限值(NaN/Inf)")
        if out["folds"]["n_fold"] > 0:
            out["errors"].append(
                f"变形场存在折叠区域(det J<=0, 占 {out['folds']['frac_fold']:.1%}), "
                "输出不可逆, 请增大 λ 或修正矛盾地标")
    return out

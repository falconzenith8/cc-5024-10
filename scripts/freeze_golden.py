"""冻结/比对古图配准的外部行为, 供重构前后做等价性比对(命令行辅助工具)。

外部行为只限定两件事(见 README):
  - 位移场: 在固定源域网格上对 f(x) 的求值;
  - 导出像素: 固定源图/目标范围/输出尺寸下重采样的 RGBA 原始字节。

数值路径与界面一致: 通过 georect.kernel.solve_registration 求解,
再用 georect.resample.resample_field 反采样。默认写出/比对
tests/golden/registration_golden.json。

用法(命令行仅辅助古图配准, 图形入口仍是 server/app.py):
  python3 scripts/freeze_golden.py            # 重新冻结
  python3 scripts/freeze_golden.py --check    # 与现有快照比对, 不一致则退出码 1
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from georect.kernel import (LandmarkConstraints, RegistrationFailure,  # noqa: E402
                            solve_registration)
from georect.resample import resample_field  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "registration_golden.json"
SCHEMA = "georect-golden-v1"

TARGET_EXTENT = (0.0, 0.0, 1000.0, 800.0)
OUT_W, OUT_H = 40, 32
DISP_NX, DISP_NY = 11, 9
LAM = 1.0


def read_csv(path: Path):
    src, dst = [], []
    for line in path.read_text().splitlines():
        t = line.strip()
        if not t or t.lower().startswith("sx"):
            continue
        v = [float(x) for x in t.split(",")]
        src.append(v[0:2])
        dst.append(v[2:4])
    return np.asarray(src), np.asarray(dst)


def displacement_grid(src_extent):
    xmin, ymin, xmax, ymax = src_extent
    xs = np.linspace(xmin, xmax, DISP_NX)
    ys = np.linspace(ymin, ymax, DISP_NY)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def scenario(src, dst, image_path, src_extent):
    constraints = LandmarkConstraints(src, dst, LAM)
    result = solve_registration(constraints, src_extent=src_extent)
    errors = list(result.errors)
    warnings = list(result.warnings)
    entry = {
        "source_points": np.asarray(src, float).tolist(),
        "target_points": np.asarray(dst, float).tolist(),
        "lambda": LAM,
        "src_extent": list(src_extent),
        "target_extent": list(TARGET_EXTENT),
        "out_width": OUT_W,
        "out_height": OUT_H,
        "errors": errors,
        "warnings": warnings,
        "has_model": isinstance(result, RegistrationFailure) is False,
        "result_id": result.result_id,
        "solve_version": result.version,
    }
    if isinstance(result, RegistrationFailure):
        return entry

    field = result.field()
    pts = displacement_grid(src_extent)
    entry["displacement_query"] = pts.tolist()
    entry["displacement"] = field.evaluate(pts).tolist()
    entry["displacement_frame"] = field.displacement(
        src_extent, DISP_NX, DISP_NY).values.tolist()

    # 与 /api/warp 相同的边界策略: tol 由目标像素尺寸显式给出, 不用隐式默认值
    px = (TARGET_EXTENT[2] - TARGET_EXTENT[0]) / OUT_W
    py = (TARGET_EXTENT[3] - TARGET_EXTENT[1]) / OUT_H
    tol = 0.01 * min(px, py)

    from PIL import Image

    img = np.asarray(Image.open(image_path).convert("RGB"))
    out = resample_field(field, img, TARGET_EXTENT, OUT_W, OUT_H, tol=tol)
    rgba = np.ascontiguousarray(out.rgba).tobytes()
    entry.update({
        "tol": tol,
        "stats": out.stats,
        "rgba_sha256": hashlib.sha256(rgba).hexdigest(),
        "rgba_base64": base64.b64encode(rgba).decode("ascii"),
        "n_outside": int(out.outside.sum()),
        "n_failed": int(out.failed.sum()),
        "resample_version": out.version,
        "resample_source_result_id": out.source_result_id,
    })
    return entry


def build():
    clean_src, clean_dst = read_csv(ROOT / "demo" / "landmarks.csv")
    bad_src, bad_dst = read_csv(ROOT / "demo" / "landmarks_with_errors.csv")

    duplicated_src = np.array(
        [[0, 0], [100, 0], [0, 100], [0.2, 0.1], [100, 100]], float)
    duplicated_dst = np.array(
        [[0, 0], [100, 0], [0, 100], [50, 50], [0, 100]], float)

    return {
        "schema": SCHEMA,
        "note": "重构前冻结: 位移场与导出 RGBA 原始字节",
        "scenarios": {
            "demo_clean": scenario(clean_src, clean_dst,
                ROOT / "demo" / "scanned.png", (0.0, 0.0, 1100.0, 850.0)),
            "demo_errors": scenario(bad_src, bad_dst,
                ROOT / "demo" / "scanned.png", (0.0, 0.0, 1100.0, 850.0)),
            "contradictory_duplicate": scenario(duplicated_src, duplicated_dst,
                ROOT / "demo" / "reference.png", (0.0, 0.0, 1000.0, 800.0)),
        },
    }


def check(golden):
    """只比对外部行为: 位移场数值与导出 RGBA 字节。"""
    current = build()
    problems = []
    for name, old in golden["scenarios"].items():
        new = current["scenarios"][name]
        if old["has_model"] != new["has_model"]:
            problems.append(f"{name}: has_model 不一致")
            continue
        if not old["has_model"]:
            if old["errors"] != new["errors"]:
                problems.append(f"{name}: 错误文本不一致")
            continue
        if old["displacement"] != new["displacement"]:
            problems.append(f"{name}: 位移场数值不一致")
        if old["rgba_sha256"] != new["rgba_sha256"]:
            problems.append(f"{name}: 导出像素不一致")
        if old["stats"] != new["stats"]:
            problems.append(f"{name}: 重采样统计不一致")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="与现有快照比对, 不写文件")
    args = ap.parse_args(argv)

    if args.check:
        if not GOLDEN.exists():
            print(f"缺少快照: {GOLDEN}", file=sys.stderr)
            return 2
        problems = check(json.loads(GOLDEN.read_text()))
        if problems:
            for p in problems:
                print("不一致:", p, file=sys.stderr)
            return 1
        print(f"一致: {GOLDEN}")
        return 0

    golden = build()
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, ensure_ascii=False, allow_nan=True) + "\n")
    print(f"wrote {GOLDEN} ({GOLDEN.stat().st_size} bytes)")
    for name, e in golden["scenarios"].items():
        print(f"  {name}: model={e['has_model']} "
              f"sha={e.get('rgba_sha256', '-')[:16]} errors={len(e['errors'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

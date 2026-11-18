"""分层与版本化结果的契约测试。

覆盖:
  - 求解结果 (RegistrationSolve / RegistrationFailure) 是明确的两类返回;
  - 版本与来源(result_id)必填, 缺失即报错, 不用默认值遮错;
  - 求解 -> 变形场 -> 反采样 的来源串一致;
  - tol 必须调用方给出, 库函数不再用隐式默认值;
  - 同一组约束多次求解得到同一 result_id(结果可复现);
  - 黄金比对: 位移场与导出 RGBA 像素与重构前逐字节一致。
"""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from georect.field import (FIELD_VERSION, DeformationField, FieldError,
                           field_from_model, legacy_field_from_model)
from georect.kernel import (LandmarkConstraintError, LandmarkConstraints,
                            RegistrationFailure, RegistrationNotSolved,
                            RegistrationSolve, solve_registration)
from georect.resample import (RESAMPLE_VERSION, ResampleError, resample_field)
from georect.tps import TPSModel, evaluate

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "registration_golden.json"


def read_csv(path):
    src, dst = [], []
    for line in path.read_text().splitlines():
        t = line.strip()
        if not t or t.lower().startswith("sx"):
            continue
        v = [float(x) for x in t.split(",")]
        src.append(v[0:2]); dst.append(v[2:4])
    return np.asarray(src), np.asarray(dst)


def demo_constraints(name="landmarks.csv"):
    src, dst = read_csv(ROOT / "demo" / name)
    return LandmarkConstraints(src=src, dst=dst, lam=1.0)


class TestSolveResult:
    def test_success_is_versioned_and_reproducible(self):
        c = demo_constraints()
        a = solve_registration(c, src_extent=(0, 0, 1100, 850))
        b = solve_registration(c, src_extent=(0, 0, 1100, 850))
        assert isinstance(a, RegistrationSolve) and a.ok
        assert a.version == "georect-solve-v1"
        assert a.result_id == b.result_id  # 同一输入 -> 同一结果 id
        assert a.result_id and len(a.result_id) == 64
        assert a.model.n == c.n

    def test_different_lambda_changes_result_id(self):
        src, dst = read_csv(ROOT / "demo" / "landmarks.csv")
        c0 = LandmarkConstraints(src, dst, 0.0)
        c1 = LandmarkConstraints(src, dst, 5.0)
        assert solve_registration(c0).result_id != solve_registration(c1).result_id

    def test_failure_is_explicit_type_without_model(self):
        src = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], float)
        dst = np.array([[0, 0], [1, 0], [0, 1], [5, 5]], float)
        r = solve_registration(LandmarkConstraints(src, dst, 0.0), dup_tol=1e-6)
        assert isinstance(r, RegistrationFailure)
        assert not r.ok
        assert r.errors and r.result_id
        assert not hasattr(r, "model")
        with pytest.raises(RegistrationNotSolved):
            r.field()

    def test_contradictory_landmarks_solve_but_report_errors(self):
        src, dst = read_csv(ROOT / "demo" / "landmarks_with_errors.csv")
        r = solve_registration(LandmarkConstraints(src, dst, 1.0),
                               src_extent=(0, 0, 1100, 850))
        assert isinstance(r, RegistrationSolve) and r.ok
        # 错误地标被诊断为折叠: 求解仍然成功, 但结果带明确错误
        assert any("折叠" in e for e in r.errors)
        assert r.diagnostics["folds"]["n_fold"] > 0

    def test_contradiction_detected_and_reported(self):
        src = np.array([[0, 0], [100, 0], [0, 100], [0.2, 0.1], [100, 100]], float)
        dst = np.array([[0, 0], [100, 0], [0, 100], [50, 50], [0, 100]], float)
        r = solve_registration(LandmarkConstraints(src, dst, 1.0),
                               src_extent=(0, 0, 100, 100))
        assert r.ok
        assert r.diagnostics["contradictions"]
        assert any("矛盾" in e for e in r.errors)

    def test_constraints_reject_bad_input_at_construction(self):
        with pytest.raises(LandmarkConstraintError):
            LandmarkConstraints([[0, 0], [1, 1]], [[0, 0], [1, 1]], 0.0)
        with pytest.raises(LandmarkConstraintError):
            LandmarkConstraints([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]], -1.0)
        with pytest.raises(LandmarkConstraintError):
            LandmarkConstraints([[0, 0], [1, 0], [np.nan, 1]],
                                [[0, 0], [1, 0], [0, 1]], 0.0)


class TestFieldVersioning:
    def test_field_binds_to_source_result(self):
        r = solve_registration(demo_constraints(), src_extent=(0, 0, 1100, 850))
        f = r.field()
        assert isinstance(f, DeformationField)
        assert f.version == FIELD_VERSION
        assert f.source_result_id == r.result_id
        np.testing.assert_allclose(f.evaluate(r.constraints.src),
                                   evaluate(r.model, r.constraints.src))

    def test_field_requires_explicit_source(self):
        r = solve_registration(demo_constraints())
        with pytest.raises(FieldError):
            field_from_model(r.model, "")
        with pytest.raises(FieldError):
            DeformationField(version="wrong-version", model=r.model,
                             source_result_id="x")

    def test_legacy_adapter_marks_fingerprint_source(self):
        r = solve_registration(demo_constraints())
        f = legacy_field_from_model(r.model)
        assert f.source_result_id.startswith("legacy-model:")
        assert f.source_result_id != r.result_id

    def test_displacement_frame_shape_and_values(self):
        r = solve_registration(demo_constraints())
        f = r.field()
        frame = f.displacement((-10, -20, 30, 40), nx=5, ny=4)
        assert frame.values.shape == (20, 2)
        assert frame.xs.shape == (5,) and frame.ys.shape == (4,)
        assert frame.points.shape == (20, 2)
        # 行 0 对应 ymin: 第三列起点
        assert frame.points[0][1] == pytest.approx(-20.0)
        np.testing.assert_allclose(frame.values, evaluate(r.model, frame.points))
        with pytest.raises(FieldError):
            f.displacement((0, 0, 0, 10), nx=5, ny=4)

    def test_grid_lines_reject_degenerate_extent(self):
        r = solve_registration(demo_constraints())
        with pytest.raises(FieldError):
            r.field().grid_lines((0, 0, 0, 0))


class TestResampleContract:
    def test_tol_is_required_not_defaulted(self):
        r = solve_registration(demo_constraints(), src_extent=(0, 0, 1100, 850))
        img = np.zeros((850, 1100, 3), dtype=np.uint8)
        with pytest.raises(TypeError):
            resample_field(r.field(), img, (0, 0, 100, 100), 40, 40)  # 缺 tol

    def test_invalid_tol_rejected(self):
        r = solve_registration(demo_constraints())
        img = np.zeros((850, 1100, 3), dtype=np.uint8)
        for bad in (0.0, -0.5, float("nan"), float("inf")):
            with pytest.raises(ResampleError):
                resample_field(r.field(), img, (0, 0, 100, 100), 40, 40, tol=bad)

    def test_result_carries_versions_and_source(self):
        r = solve_registration(demo_constraints(), src_extent=(0, 0, 1100, 850))
        img = np.zeros((850, 1100, 3), dtype=np.uint8)
        out = resample_field(r.field(), img, (0, 0, 1100, 850), 40, 32, tol=0.275)
        assert out.version == RESAMPLE_VERSION
        assert out.field_version == FIELD_VERSION
        assert out.source_result_id == r.result_id
        assert out.rgba.shape == (32, 40, 4)
        assert out.rgba.dtype == np.uint8

    def test_legacy_dict_shape_matches_old_api(self):
        r = solve_registration(demo_constraints(), src_extent=(0, 0, 1100, 850))
        img = np.zeros((850, 1100, 3), dtype=np.uint8)
        legacy = resample_field(r.field(), img, (0, 0, 1100, 850), 40, 32,
                                tol=0.275).to_legacy_dict()
        assert set(legacy) == {"rgba", "valid", "outside", "failed", "reason", "stats"}


class TestLegacyEntrypointsStillUsable:
    """古图配准旧样例: 只用重构前的入口也要能运行到底。"""

    def test_legacy_demo_pipeline_runs_end_to_end(self):
        from PIL import Image

        from georect.diagnostics import run_all
        from georect.export import model_from_dict, model_to_dict
        from georect.warp import resample_to_grid

        src, dst = read_csv(ROOT / "demo" / "landmarks.csv")
        diag = run_all(src, dst, 1.0, src_extent=(0, 0, 1100, 850))
        assert not diag["errors"], diag["errors"]
        model = diag["model"]

        # 导出/导入参数后结果可复现
        d = model_to_dict(model, dst)
        model2 = model_from_dict(json.loads(json.dumps(d)))
        probe = np.array([[10.0, 20.0], [700.0, 300.0], [1050.0, 800.0]])
        np.testing.assert_allclose(evaluate(model2, probe), evaluate(model, probe))

        img = np.asarray(Image.open(ROOT / "demo" / "scanned.png").convert("RGB"))
        out = resample_to_grid(model, img, (0, 0, 1000, 800), 120, 96)
        assert set(out) == {"rgba", "valid", "outside", "failed", "reason", "stats"}
        assert out["stats"]["n_valid"] > 0
        assert out["stats"]["tol"] > 0  # 旧入口的隐式容差政策保持不变
        assert out["rgba"].shape == (96, 120, 4)

    def test_export_dict_is_backward_compatible(self):
        from georect.export import solve_to_dict

        r = solve_registration(demo_constraints(), src_extent=(0, 0, 1100, 850))
        d = solve_to_dict(r)
        assert d["format"] == "georect-tps-params"
        assert d["kernel"] == "tps-r2logr"
        assert d["version"] == "1.1"
        assert d["solve"] == {"version": "georect-solve-v1", "result_id": r.result_id}
        # 既有字段仍在, 能被旧导入器读取
        for key in ("source_points", "target_points", "weights", "affine", "lambda",
                    "normalize"):
            assert key in d

    def test_warp_adapter_rejects_non_model(self):
        from georect.warp import resample_to_grid

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(AttributeError):
            resample_to_grid(object(), img, (0, 0, 10, 10), 32, 32)


class TestGoldenEquivalence:
    """位移场与导出像素: 重构前后必须逐字节一致。"""

    @pytest.fixture(scope="class")
    @classmethod
    def golden(cls):
        return json.loads(GOLDEN.read_text())

    @pytest.mark.parametrize("name", ["demo_clean", "demo_errors", "contradictory_duplicate"])
    def test_old_entrypoint_matches_golden(self, golden, name):
        from georect.diagnostics import run_all
        from georect.warp import resample_to_grid
        from PIL import Image

        g = golden["scenarios"][name]
        diag = run_all(np.asarray(g["source_points"]), np.asarray(g["target_points"]),
                       g["lambda"], src_extent=g["src_extent"])
        assert ("model" in diag) == g["has_model"]
        assert list(diag["errors"]) == g["errors"]
        if not g["has_model"]:
            return
        model = diag["model"]
        disp = evaluate(model, np.asarray(g["displacement_query"]))
        np.testing.assert_array_equal(disp, np.asarray(g["displacement"]))
        img_path = (ROOT / "demo" / "scanned.png" if name.startswith("demo")
                    else ROOT / "demo" / "reference.png")
        img = np.asarray(Image.open(img_path).convert("RGB"))
        out = resample_to_grid(model, img, tuple(g["target_extent"]),
                               g["out_width"], g["out_height"], tol=g["tol"])
        sha = hashlib.sha256(np.ascontiguousarray(out["rgba"]).tobytes()).hexdigest()
        assert sha == g["rgba_sha256"]
        assert out["stats"] == g["stats"]

    @pytest.mark.parametrize("name", ["demo_clean", "demo_errors", "contradictory_duplicate"])
    def test_new_layered_path_matches_golden(self, golden, name):
        from PIL import Image

        g = golden["scenarios"][name]
        c = LandmarkConstraints(np.asarray(g["source_points"]),
                                np.asarray(g["target_points"]), g["lambda"])
        result = solve_registration(c, src_extent=tuple(g["src_extent"]))
        assert result.ok == g["has_model"]
        if not result.ok:
            assert list(result.errors) == g["errors"]
            return
        field = result.field()
        disp = field.evaluate(np.asarray(g["displacement_query"]))
        np.testing.assert_array_equal(disp, np.asarray(g["displacement"]))
        img_path = (ROOT / "demo" / "scanned.png" if name.startswith("demo")
                    else ROOT / "demo" / "reference.png")
        img = np.asarray(Image.open(img_path).convert("RGB"))
        out = resample_field(field, img, tuple(g["target_extent"]),
                             g["out_width"], g["out_height"], tol=g["tol"])
        sha = hashlib.sha256(np.ascontiguousarray(out.rgba).tobytes()).hexdigest()
        assert sha == g["rgba_sha256"]
        assert out.stats == g["stats"]
        # 黄金文件的 base64 也能还原成同一像素
        assert base64.b64encode(np.ascontiguousarray(out.rgba).tobytes()).decode("ascii") \
            == g["rgba_base64"]

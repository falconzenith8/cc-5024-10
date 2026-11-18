import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app as flask_app  # noqa: E402


@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _png_bytes(w=64, h=48):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.arange(w, dtype=np.uint8)[None, :] * 4
    arr[..., 1] = np.arange(h, dtype=np.uint8)[:, None] * 5
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200


def test_fit_and_warp_roundtrip(client):
    src = [[100, 100], [500, 120], [120, 480], [480, 460], [300, 300]]
    dst = [[x + 10, y - 5] for x, y in src]
    payload = {
        "source_points": src,
        "target_points": dst,
        "lambda": 0.0,
        "target_extent": [0, 0, 600, 600],
        "src_extent": [0, 0, 600, 600],
    }
    r = client.post("/api/fit", json=payload)
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"], d["errors"]
    for key in ("model", "uncertainty", "grid_lines", "rms_error", "sigma_hat",
                "folds", "loocv"):
        assert key in d
    assert d["rms_error"] < 1e-6
    assert d["folds"]["n_fold"] == 0

    up = client.post("/api/upload/source",
                     data={"file": (io.BytesIO(_png_bytes()), "s.png")},
                     content_type="multipart/form-data")
    assert up.get_json()["ok"]
    sid = up.get_json()["id"]

    w = client.post("/api/warp", json={
        "source_id": sid,
        "source_points": src,
        "target_points": dst,
        "lambda": 0.0,
        "target_extent": [0, 0, 600, 600],
        "out_width": 120,
        "out_height": 120,
    })
    wd = w.get_json()
    assert wd["ok"], wd.get("errors")
    png = Image.open(io.BytesIO(base64.b64decode(wd["png_base64"])))
    assert png.size == (120, 120) and png.mode == "RGBA"
    assert wd["stats"]["n_valid"] > 0


def test_fit_and_warp_share_one_versioned_result(client):
    """预览(/api/fit)与重采样(/api/warp)必须读取同一个版本化求解结果。"""
    src = [[100, 100], [500, 120], [120, 480], [480, 460], [300, 300]]
    dst = [[x + 10, y - 5] for x, y in src]
    payload = {
        "source_points": src,
        "target_points": dst,
        "lambda": 0.0,
        "target_extent": [0, 0, 600, 600],
    }
    fit = client.post("/api/fit", json=dict(payload, src_extent=[0, 0, 600, 600])).get_json()
    assert fit["ok"], fit["errors"]
    assert fit["solve"]["version"] == "georect-solve-v1"
    rid = fit["solve"]["result_id"]
    assert fit["model"]["solve"]["result_id"] == rid  # 导出读取同一结果

    # 上传 600x600 源图 -> /api/warp 的 src_extent 与 fit 相同 -> 同一 result_id
    up = client.post("/api/upload/source",
                     data={"file": (io.BytesIO(_png_bytes(600, 600)), "s.png")},
                     content_type="multipart/form-data")
    sid = up.get_json()["id"]
    warp = client.post("/api/warp", json=dict(
        payload, source_id=sid, out_width=64, out_height=64)).get_json()
    assert warp["ok"], warp.get("errors")
    assert warp["solve"]["version"] == "georect-solve-v1"
    assert warp["solve"]["result_id"] == rid
    assert warp["resample"]["source_result_id"] == rid
    assert warp["resample"]["tol"] > 0

    # 同一组地标但 λ 不同 -> 结果 id 必须不同
    lam3 = dict(payload, source_id=sid, out_width=64, out_height=64)
    lam3["lambda"] = 3.0
    warp_lam = client.post("/api/warp", json=lam3).get_json()
    assert warp_lam["ok"]
    assert warp_lam["solve"]["result_id"] != rid


def test_fit_legacy_fields_preserved(client):
    src = [[100, 100], [500, 120], [120, 480], [480, 460], [300, 300]]
    dst = [[x + 10, y - 5] for x, y in src]
    d = client.post("/api/fit", json={
        "source_points": src, "target_points": dst, "lambda": 0.0,
        "target_extent": [0, 0, 600, 600], "src_extent": [0, 0, 600, 600],
    }).get_json()
    for key in ("ok", "errors", "warnings", "model", "fitted", "fit_residuals",
                "loocv", "loocv_flags", "rms_error", "max_error", "loocv_rms",
                "sigma_hat", "bending_energy", "folds", "uncertainty", "hull",
                "grid_lines", "duplicates_src", "duplicates_dst",
                "contradictions", "geometry", "condition_number"):
        assert key in d, key
    assert d["model"]["kernel"] == "tps-r2logr"
    assert d["model"]["solve"]["result_id"] == d["solve"]["result_id"]


def test_fit_returns_typed_errors_not_500(client):
    r = client.post("/api/fit", json={"source_points": "nope"})
    assert r.status_code == 400
    assert not r.get_json()["ok"]
    r2 = client.post("/api/fit", data="not json", content_type="text/plain")
    assert r2.status_code == 400
    r3 = client.post("/api/warp", json={
        "source_id": "missing", "source_points": [[0, 0], [1, 0], [0, 1]],
        "target_points": [[0, 0], [1, 0], [0, 1]], "lambda": 0,
        "target_extent": [0, 0, 10, 10], "out_width": 32, "out_height": 32})
    assert r3.status_code == 400
    assert "source_id" in r3.get_json()["errors"][0]


def test_three_landmarks_solve_without_crash(client):
    """3 对地标是合法输入: 应成功求解, 且响应必须是合法 JSON(无裸 NaN)。"""
    payload = {
        "source_points": [[0, 0], [10, 0], [0, 10]],
        "target_points": [[5, 5], [15, 0], [0, 10]],
        "lambda": 0.0,
        "target_extent": [0, 0, 10, 10],
        "src_extent": [0, 0, 10, 10],
    }
    r = client.post("/api/fit", json=payload)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "NaN" not in body and "Infinity" not in body
    d = r.get_json()
    assert d["ok"], d["errors"]
    assert d["loocv"] == [[None, None]] * 3  # LOOCV 无定义 -> 明确的 null
    assert d["loocv_rms"] is None
    assert d["sigma_hat"] is None


def test_fit_rejects_bad_input(client):
    r = client.post("/api/fit", json={
        "source_points": [[0, 0], [1, 1]],
        "target_points": [[0, 0], [1, 1]],
        "lambda": 0.0,
        "target_extent": [0, 0, 10, 10],
    })
    d = r.get_json()
    assert not d["ok"]
    assert d["errors"]

    r2 = client.post("/api/fit", json={
        "source_points": [[0, 0], [10, 0], [0, 10], [0, 0]],
        "target_points": [[0, 0], [10, 0], [0, 10], [9, 9]],
        "lambda": 0.0,
        "target_extent": [0, 0, 10, 10],
    })
    d2 = r2.get_json()
    assert not d2["ok"]
    assert any("矛盾" in e or "奇异" in e for e in d2["errors"])

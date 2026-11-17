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

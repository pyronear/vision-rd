import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from data_quality_frame_level.review_app.main import create_app
from data_quality_frame_level.review_app.state import Paths


@pytest.fixture
def app_tree(tmp_path: Path) -> tuple[TestClient, Paths]:
    split = tmp_path / "01_raw" / "datasets" / "val"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir(parents=True)
    stems = ["s_2024-01-01T00-00-00", "s_2024-01-01T00-00-30"]
    for st in stems:
        (split / "images" / f"{st}.jpg").write_bytes(b"jpeg")
        (split / "labels" / f"{st}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    pred_path = tmp_path / "07_model_output" / "m" / "val" / "predictions.json"
    pred_path.parent.mkdir(parents=True)
    pred_path.write_text(
        json.dumps(
            {
                "model_name": "m",
                "split_dir": "data/01_raw/datasets/val",
                "conf_thresh": 0.05,
                "frames": {
                    stems[0]: {
                        "image_path": f"images/{stems[0]}.jpg",
                        "predictions": [
                            {
                                "class_id": 0,
                                "cx": 0.7,
                                "cy": 0.7,
                                "w": 0.1,
                                "h": 0.1,
                                "conf": 0.9,
                            }
                        ],
                    },
                    stems[1]: {
                        "image_path": f"images/{stems[1]}.jpg",
                        "predictions": [],
                    },
                },
            }
        )
    )
    paths = Paths(
        split_dir=split,
        predictions_path=pred_path,
        review_path=tmp_path / "09_review" / "m" / "val" / "review.json",
    )
    app = create_app(
        contexts={("m", "val"): paths},
        models=["m"],
        splits=["val"],
        repo_root=tmp_path,
    )
    return TestClient(app), paths


def test_get_contexts(app_tree):
    client, _ = app_tree
    r = client.get("/api/contexts")
    assert r.status_code == 200
    body = r.json()
    assert body["models"] == ["m"]
    assert body["splits"] == ["val"]


def test_get_queue_fp(app_tree):
    client, _ = app_tree
    r = client.get(
        "/api/queue",
        params={
            "model": "m",
            "split": "val",
            "view": "fp",
            "conf": 0.05,
            "iou": 0.05,
            "review_conf": 0.5,
        },
    )
    assert r.status_code == 200
    items = r.json()["items"]
    by_stem = {i["stem"]: i for i in items}
    assert by_stem["s_2024-01-01T00-00-00"]["kind"] == "fp"
    assert by_stem["s_2024-01-01T00-00-30"]["kind"] == "none"
    assert [i["stem"] for i in items] == [
        "s_2024-01-01T00-00-00",
        "s_2024-01-01T00-00-30",
    ]


def test_get_sample_returns_layers_and_neighbors(app_tree):
    client, _ = app_tree
    r = client.get(
        "/api/sample",
        params={
            "model": "m",
            "split": "val",
            "stem": "s_2024-01-01T00-00-00",
            "conf": 0.05,
            "iou": 0.05,
            "review_conf": 0.5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["original_gt"]) == 1
    assert len(body["predictions"]) == 1
    assert body["sequence_neighbors"][0]["stem"] in {
        "s_2024-01-01T00-00-00",
        "s_2024-01-01T00-00-30",
    }


def test_post_sample_persists(app_tree):
    client, paths = app_tree
    r = client.post(
        "/api/sample",
        params={"model": "m", "split": "val"},
        json={
            "stem": "s_2024-01-01T00-00-00",
            "status": "reviewed",
            "bboxes": [{"class_id": 0, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}],
            "reviewer": "arthur",
            "note": "fixed",
        },
    )
    assert r.status_code == 200
    payload = json.loads(paths.review_path.read_text())
    sample = payload["samples"]["s_2024-01-01T00-00-00"]
    assert sample["status"] == "reviewed"
    assert sample["bboxes"][0]["cx"] == 0.4


def test_get_contexts_includes_dvc_warnings(app_tree):
    client, paths = app_tree
    dvc_path = paths.review_path.with_suffix(paths.review_path.suffix + ".dvc")
    dvc_path.parent.mkdir(parents=True, exist_ok=True)
    dvc_path.write_text(
        "outs:\n- md5: ffeeddccbbaa\n  size: 0\n  hash: md5\n  path: review.json\n"
    )
    client.post(
        "/api/sample",
        params={"model": "m", "split": "val"},
        json={
            "stem": "s_2024-01-01T00-00-00",
            "status": "reviewed",
            "bboxes": [],
            "reviewer": "arthur",
        },
    )
    body = client.get("/api/contexts").json()
    assert "dvc_warnings" in body
    assert len(body["dvc_warnings"]) == 1
    w = body["dvc_warnings"][0]
    assert w["model"] == "m"
    assert w["split"] == "val"
    assert w["kind"] == "stale_local"

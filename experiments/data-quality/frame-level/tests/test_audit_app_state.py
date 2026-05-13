import json
from pathlib import Path

import pytest

from data_quality_frame_level.audit_app.state import AppState, Paths
from data_quality_frame_level.dataset import BBox


def _write_predictions(p: Path, frames: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "model_name": "m",
                "split_dir": "data/01_raw/datasets/val",
                "conf_thresh": 0.05,
                "frames": frames,
            }
        )
    )


@pytest.fixture
def fake_tree(tmp_path: Path) -> Paths:
    split = tmp_path / "01_raw" / "datasets" / "val"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir(parents=True)
    stem = "seq_2024-01-01T00-00-00"
    (split / "images" / f"{stem}.jpg").write_bytes(b"jpeg")
    (split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    pred_path = tmp_path / "07_model_output" / "m" / "val" / "predictions.json"
    _write_predictions(
        pred_path,
        {
            stem: {
                "image_path": f"images/{stem}.jpg",
                "predictions": [
                    {
                        "class_id": 0,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.1,
                        "h": 0.1,
                        "conf": 0.9,
                    }
                ],
            }
        },
    )
    return Paths(
        split_dir=split,
        predictions_path=pred_path,
        review_path=tmp_path / "09_review" / "m" / "val" / "review.json",
    )


def test_load_populates_predictions_and_gt(fake_tree: Paths):
    state = AppState.load(model="m", split="val", paths=fake_tree)
    assert "seq_2024-01-01T00-00-00" in state.predictions
    assert state.predictions["seq_2024-01-01T00-00-00"][0].conf == 0.9
    assert state.gt["seq_2024-01-01T00-00-00"][0].cx == 0.5


def test_save_sample_writes_review_json(fake_tree: Paths):
    state = AppState.load(model="m", split="val", paths=fake_tree)
    state.save_sample(
        stem="seq_2024-01-01T00-00-00",
        status="reviewed",
        bboxes=[BBox(class_id=0, cx=0.4, cy=0.4, w=0.2, h=0.2)],
        spurious_originals=[],
        reviewer="arthur",
        note="moved",
    )
    payload = json.loads(fake_tree.review_path.read_text())
    sample = payload["samples"]["seq_2024-01-01T00-00-00"]
    assert sample["status"] == "reviewed"
    assert sample["bboxes"][0]["cx"] == 0.4
    assert sample["reviewer"] == "arthur"


def test_delete_sample_removes_entry(fake_tree: Paths):
    state = AppState.load(model="m", split="val", paths=fake_tree)
    state.save_sample(
        stem="seq_2024-01-01T00-00-00",
        status="reviewed",
        bboxes=[],
        spurious_originals=[],
        reviewer=None,
        note=None,
    )
    state.delete_sample(stem="seq_2024-01-01T00-00-00")
    assert "seq_2024-01-01T00-00-00" not in state.review.samples
    payload = json.loads(fake_tree.review_path.read_text())
    assert "seq_2024-01-01T00-00-00" not in payload["samples"]


def test_delete_sample_unknown_stem_is_noop(fake_tree: Paths):
    state = AppState.load(model="m", split="val", paths=fake_tree)
    state.delete_sample(stem="seq_2024-01-01T00-00-00")
    assert state.review.samples == {}

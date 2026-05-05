import json
from pathlib import Path

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.persistence import (
    ReviewState,
    SampleReview,
    read_review_state,
    write_review_state,
)


def _bb(cx, cy, w=0.1, h=0.1):
    return BBox(class_id=0, cx=cx, cy=cy, w=w, h=h)


def test_read_missing_file_returns_empty(tmp_path: Path):
    state = read_review_state(tmp_path / "review.json", model="m", split="val")
    assert state.samples == {}
    assert state.model == "m"
    assert state.split == "val"


def test_round_trip(tmp_path: Path):
    p = tmp_path / "review.json"
    state = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(
                status="reviewed",
                bboxes=[_bb(0.5, 0.5)],
                reviewer="arthur",
                note="ok",
                reviewed_at="2026-05-05T14:00:00Z",
            )
        },
    )
    write_review_state(p, state)
    payload = json.loads(p.read_text())
    assert payload["version"] == 1
    assert payload["model_name"] == "m"
    assert payload["split"] == "val"
    assert "stem_a" in payload["samples"]
    reloaded = read_review_state(p, model="m", split="val")
    assert reloaded == state


def test_write_is_atomic(tmp_path: Path):
    p = tmp_path / "review.json"
    write_review_state(p, ReviewState(model="m", split="val", samples={}))
    assert p.exists()
    assert not (tmp_path / "review.json.tmp").exists()


def test_serialization_is_sorted(tmp_path: Path):
    p = tmp_path / "review.json"
    state = ReviewState(
        model="m",
        split="val",
        samples={
            "z": SampleReview(status="reviewed", bboxes=[]),
            "a": SampleReview(status="reviewed", bboxes=[]),
        },
    )
    write_review_state(p, state)
    text = p.read_text()
    assert text.index('"a"') < text.index('"z"')

from pathlib import Path

from bbox_tube_temporal.types import Detection, FrameDetections
from pyrocore import Frame

from tube_builder_lab.cache import cache_one
from tube_builder_lab.detections_io import read_detections


def test_cache_one_writes_roundtrippable_detections(tmp_path: Path):
    frames = [
        Frame(frame_id="a", image_path=tmp_path / "a.jpg", timestamp=None),
        Frame(frame_id="b", image_path=tmp_path / "b.jpg", timestamp=None),
    ]
    canned = [
        FrameDetections(
            frame_idx=0,
            frame_id="a",
            timestamp=None,
            detections=[
                Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.8)
            ],
        ),
        FrameDetections(frame_idx=1, frame_id="b", timestamp=None, detections=[]),
    ]
    out_dir = tmp_path / "detections"

    def fake_run_yolo(fs: list[Frame]) -> list[FrameDetections]:
        assert fs == frames
        return canned

    path = cache_one(
        out_dir=out_dir, key="platform_42", frames=frames, run_yolo=fake_run_yolo
    )
    assert path == out_dir / "platform_42.json"
    assert read_detections(path) == canned


def test_cache_one_skips_when_present(tmp_path: Path):
    out_dir = tmp_path / "detections"
    out_dir.mkdir()
    (out_dir / "platform_9.json").write_text("[]")
    calls = {"n": 0}

    def run_yolo(_):
        calls["n"] += 1
        return []

    cache_one(
        out_dir=out_dir,
        key="platform_9",
        frames=[],
        run_yolo=run_yolo,
        overwrite=False,
    )
    assert calls["n"] == 0  # skipped, did not run YOLO

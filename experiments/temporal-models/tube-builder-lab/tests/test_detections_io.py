from pathlib import Path

from bbox_tube_temporal.types import Detection, FrameDetections

from tube_builder_lab.detections_io import read_detections, write_detections


def test_detections_roundtrip(tmp_path: Path):
    fds = [
        FrameDetections(
            frame_idx=0,
            frame_id="a",
            timestamp=None,
            detections=[
                Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.2, confidence=0.9),
                Detection(class_id=0, cx=0.2, cy=0.3, w=0.05, h=0.05, confidence=0.4),
            ],
        ),
        FrameDetections(frame_idx=1, frame_id="b", timestamp=None, detections=[]),
    ]
    path = tmp_path / "platform_42.json"
    write_detections(path, fds)
    got = read_detections(path)
    assert got == fds

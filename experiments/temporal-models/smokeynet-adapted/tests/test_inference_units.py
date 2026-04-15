"""Pure-function unit tests for inference helpers."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pyrocore.types import Frame

from smokeynet_adapted.inference import filter_and_interpolate_tubes, run_yolo_on_frames
from smokeynet_adapted.types import Detection, FrameDetections, Tube, TubeEntry


def _fake_yolo_result(
    xywhn: list[tuple[float, float, float, float, float]],
) -> MagicMock:
    """Build a MagicMock shaped like ultralytics Results for one image.

    xywhn: list of (cx, cy, w, h, conf) tuples.
    """
    boxes = MagicMock()
    boxes.__len__ = lambda self: len(xywhn)
    boxes.xywhn = torch.tensor([[c, cy, w, h] for (c, cy, w, h, _) in xywhn])
    boxes.conf = torch.tensor([conf for (_, _, _, _, conf) in xywhn])
    boxes.cls = torch.zeros(len(xywhn))
    result = MagicMock()
    result.boxes = boxes
    return result


class TestRunYoloOnFrames:
    def test_batches_all_frames_in_single_call(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = [_fake_yolo_result([]), _fake_yolo_result([])]
        frames = [
            Frame(frame_id="f0", image_path=Path("/x/f0.jpg"), timestamp=None),
            Frame(frame_id="f1", image_path=Path("/x/f1.jpg"), timestamp=None),
        ]

        run_yolo_on_frames(
            yolo, frames, confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )

        assert yolo.predict.call_count == 1
        args, kwargs = yolo.predict.call_args
        assert args[0] == ["/x/f0.jpg", "/x/f1.jpg"]
        assert kwargs["conf"] == 0.01
        assert kwargs["iou"] == 0.2
        assert kwargs["imgsz"] == 1024
        assert kwargs["verbose"] is False

    def test_converts_detections(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = [
            _fake_yolo_result([(0.5, 0.4, 0.1, 0.2, 0.9)]),
            _fake_yolo_result([]),
        ]
        ts = datetime(2024, 1, 1, 12, 0, 0)
        frames = [
            Frame(frame_id="f0", image_path=Path("/x/f0.jpg"), timestamp=ts),
            Frame(frame_id="f1", image_path=Path("/x/f1.jpg"), timestamp=None),
        ]

        result = run_yolo_on_frames(
            yolo, frames, confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )

        assert len(result) == 2

        # Frame 0: one detection, check structural + numeric approximately.
        fd0 = result[0]
        assert fd0.frame_idx == 0
        assert fd0.frame_id == "f0"
        assert fd0.timestamp == ts
        assert len(fd0.detections) == 1
        d0 = fd0.detections[0]
        assert d0.class_id == 0
        assert d0.cx == pytest.approx(0.5, rel=1e-6)
        assert d0.cy == pytest.approx(0.4, rel=1e-6)
        assert d0.w == pytest.approx(0.1, rel=1e-6)
        assert d0.h == pytest.approx(0.2, rel=1e-6)
        assert d0.confidence == pytest.approx(0.9, rel=1e-6)

        # Frame 1: no detections.
        assert result[1] == FrameDetections(
            frame_idx=1, frame_id="f1", timestamp=None, detections=[]
        )

    def test_empty_frames_returns_empty(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = []
        result = run_yolo_on_frames(
            yolo, [], confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )
        assert result == []
        yolo.predict.assert_not_called()


def _tube(tid: int, entries: list[tuple[int, Detection | None]]) -> Tube:
    return Tube(
        tube_id=tid,
        entries=[TubeEntry(frame_idx=i, detection=d) for (i, d) in entries],
        start_frame=entries[0][0],
        end_frame=entries[-1][0],
    )


def _det(cx: float = 0.5, cy: float = 0.5, w: float = 0.1, h: float = 0.1) -> Detection:
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9)


class TestFilterAndInterpolate:
    def test_drops_tubes_shorter_than_min_length(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, _det())]),             # length 2 - keep
            _tube(1, [(3, _det())]),                           # length 1 - drop
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=1, interpolate_gaps=False
        )
        assert [t.tube_id for t in out] == [0]

    def test_drops_tubes_with_too_few_observed(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, None), (2, None), (3, None)]),
            _tube(1, [(0, _det()), (1, _det()), (2, None), (3, None)]),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=False
        )
        assert [t.tube_id for t in out] == [1]

    def test_interpolation_applied_when_enabled(self) -> None:
        tubes = [
            _tube(
                0,
                [
                    (0, _det(cx=0.2)),
                    (1, None),
                    (2, _det(cx=0.4)),
                ],
            ),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=True
        )
        assert len(out) == 1
        mid = out[0].entries[1]
        assert mid.is_gap is True
        assert mid.detection is not None
        assert mid.detection.cx == pytest.approx(0.3)

    def test_interpolation_skipped_when_disabled(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, None), (2, _det())]),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=False
        )
        assert out[0].entries[1].detection is None

    def test_empty_input(self) -> None:
        assert filter_and_interpolate_tubes(
            [], min_tube_length=2, min_detected_entries=1, interpolate_gaps=True
        ) == []

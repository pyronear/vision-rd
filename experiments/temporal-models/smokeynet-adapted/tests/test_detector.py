"""Tests for smokeynet_adapted.detector.

YOLO model is mocked to avoid requiring GPU / model weights in CI.
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import torch

from smokeynet_adapted.detector import run_yolo_on_frame, run_yolo_on_sequence


def _make_mock_result(boxes_data: list[tuple[float, float, float, float, float, int]]):
    """Build a mock YOLO result with given (cx, cy, w, h, conf, cls) entries."""

    @dataclass
    class MockBoxes:
        xywhn: torch.Tensor
        conf: torch.Tensor
        cls: torch.Tensor

        def __len__(self):
            return len(self.xywhn)

    if not boxes_data:
        result = MagicMock()
        result.boxes = None
        return result

    xywhn = torch.tensor([[cx, cy, w, h] for cx, cy, w, h, _, _ in boxes_data])
    conf = torch.tensor([c for _, _, _, _, c, _ in boxes_data])
    cls = torch.tensor([cl for _, _, _, _, _, cl in boxes_data], dtype=torch.int32)

    result = MagicMock()
    result.boxes = MockBoxes(xywhn=xywhn, conf=conf, cls=cls)
    return result


class TestRunYoloOnFrame:
    def test_no_detections(self):
        model = MagicMock()
        model.predict.return_value = [_make_mock_result([])]
        dets = run_yolo_on_frame(model, Path("fake.jpg"))
        assert dets == []

    def test_single_detection(self):
        model = MagicMock()
        model.predict.return_value = [_make_mock_result([(0.5, 0.5, 0.1, 0.1, 0.9, 0)])]
        dets = run_yolo_on_frame(model, Path("fake.jpg"))
        assert len(dets) == 1
        assert dets[0].cx == 0.5
        assert abs(dets[0].confidence - 0.9) < 1e-6
        assert dets[0].class_id == 0

    def test_multiple_detections(self):
        model = MagicMock()
        model.predict.return_value = [
            _make_mock_result(
                [
                    (0.3, 0.3, 0.1, 0.1, 0.8, 0),
                    (0.7, 0.7, 0.2, 0.2, 0.6, 0),
                ]
            )
        ]
        dets = run_yolo_on_frame(model, Path("fake.jpg"))
        assert len(dets) == 2


class TestRunYoloOnSequence:
    def test_sequence(self):
        model = MagicMock()
        model.predict.return_value = [_make_mock_result([(0.5, 0.5, 0.1, 0.1, 0.9, 0)])]
        results = run_yolo_on_sequence(
            model,
            frame_paths=[Path("a.jpg"), Path("b.jpg")],
            frame_ids=["a", "b"],
            timestamps=[None, None],
        )
        assert len(results) == 2
        assert results[0].frame_idx == 0
        assert results[1].frame_idx == 1

    def test_max_detections_per_frame(self):
        model = MagicMock()
        model.predict.return_value = [
            _make_mock_result(
                [
                    (0.3, 0.3, 0.1, 0.1, 0.5, 0),
                    (0.5, 0.5, 0.1, 0.1, 0.9, 0),
                    (0.7, 0.7, 0.1, 0.1, 0.7, 0),
                ]
            )
        ]
        results = run_yolo_on_sequence(
            model,
            frame_paths=[Path("a.jpg")],
            frame_ids=["a"],
            timestamps=[None],
            max_detections_per_frame=2,
        )
        assert len(results[0].detections) == 2
        # Should keep top-2 by confidence: 0.9 and 0.7
        confs = [d.confidence for d in results[0].detections]
        assert abs(confs[0] - 0.9) < 1e-6
        assert abs(confs[1] - 0.7) < 1e-6

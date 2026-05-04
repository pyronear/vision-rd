"""Tests for FireTubeModel (TemporalModel implementation)."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
from pyrocore import Frame, TemporalModelOutput

from src.classifier import train_classifier
from src.model import FireTubeModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INFER_PARAMS = {
    "confidence_threshold": 0.01,
    "iou_nms": 0.2,
    "image_size": 1024,
}

TUBE_PARAMS = {
    "crop_size": 32,
    "max_tube_length": 50,
    "confidence_threshold": 0.3,
    "max_detection_area": 0.05,
    "iou_threshold": 0.1,
}


def _make_yolo_prediction(
    detections: list[tuple[int, float, float, float, float, float]],
):
    """Build a mock YOLO prediction result."""
    if not detections:
        pred = MagicMock()
        pred.boxes = MagicMock()
        pred.boxes.__len__ = lambda self: 0
        pred.boxes.__iter__ = lambda self: iter([])
        return [pred]

    pred = MagicMock()
    boxes = pred.boxes
    n = len(detections)
    boxes.__len__ = lambda self, _n=n: _n

    xywhn_tensors = []
    cls_tensors = []
    conf_tensors = []
    for class_id, cx, cy, w, h, conf in detections:
        t = MagicMock()
        t.tolist.return_value = [cx, cy, w, h]
        xywhn_tensors.append(t)
        c = MagicMock()
        c.item.return_value = class_id
        cls_tensors.append(c)
        cf = MagicMock()
        cf.item.return_value = conf
        conf_tensors.append(cf)

    boxes.xywhn = xywhn_tensors
    boxes.cls = cls_tensors
    boxes.conf = conf_tensors
    return [pred]


def _make_frames(n: int, tmp_path: Path) -> list[Frame]:
    """Create dummy Frame objects with real JPEG images."""
    frames = []
    for i in range(n):
        p = tmp_path / f"frame_{i:03d}.jpg"
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        img.save(p)
        frames.append(
            Frame(
                frame_id=f"frame_{i:03d}",
                image_path=p,
                timestamp=datetime(2024, 1, 1, 12, 0, i),
            )
        )
    return frames


def _make_dummy_classifier():
    """Train a small RF on random data for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 24)
    y = np.array([0] * 10 + [1] * 10)
    return train_classifier(X, y, n_estimators=5, max_depth=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_direct_init(self) -> None:
        model = FireTubeModel(
            yolo_model=MagicMock(),
            classifier=_make_dummy_classifier(),
            infer_params=INFER_PARAMS,
            tube_params=TUBE_PARAMS,
        )
        assert isinstance(model, FireTubeModel)


class TestPredictNoDetections:
    def test_returns_negative(self, tmp_path: Path) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([])

        model = FireTubeModel(
            yolo_model=yolo,
            classifier=_make_dummy_classifier(),
            infer_params=INFER_PARAMS,
            tube_params=TUBE_PARAMS,
        )
        frames = _make_frames(5, tmp_path)
        output = model.predict(frames)

        assert isinstance(output, TemporalModelOutput)
        assert output.is_positive is False
        assert output.trigger_frame_index is None
        assert output.details["num_tubes"] == 0


class TestPredictWithDetections:
    def test_produces_tubes(self, tmp_path: Path) -> None:
        det = (0, 0.5, 0.5, 0.1, 0.1, 0.8)
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([det])

        model = FireTubeModel(
            yolo_model=yolo,
            classifier=_make_dummy_classifier(),
            infer_params=INFER_PARAMS,
            tube_params=TUBE_PARAMS,
        )
        frames = _make_frames(5, tmp_path)
        output = model.predict(frames)

        assert isinstance(output, TemporalModelOutput)
        assert output.details["num_tubes"] >= 1


class TestOutputDetails:
    def test_details_keys(self, tmp_path: Path) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([])

        model = FireTubeModel(
            yolo_model=yolo,
            classifier=_make_dummy_classifier(),
            infer_params=INFER_PARAMS,
            tube_params=TUBE_PARAMS,
        )
        output = model.predict(_make_frames(5, tmp_path))

        assert "num_tubes" in output.details
        assert "num_positive_tubes" in output.details
        assert "num_detections_total" in output.details
        assert "original_sequence_length" in output.details
        assert "padded_sequence_length" in output.details
        assert "tube_lengths" in output.details

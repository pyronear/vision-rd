"""Tests for FsmTrackingModel (TemporalModel implementation)."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyrocore import Frame, TemporalModelOutput

from src.model import FsmTrackingModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INFER_PARAMS = {
    "confidence_threshold": 0.01,
    "iou_nms": 0.2,
    "image_size": 1024,
}

PREFILTER_PARAMS = {
    "confidence_threshold": 0.3,
    "max_detection_area": 0.05,
}

TRACKER_PARAMS = {
    "iou_threshold": 0.1,
    "min_consecutive": 3,
    "max_misses": 0,
    "use_confidence_filter": False,
    "min_mean_confidence": 0.3,
    "use_area_change_filter": False,
    "min_area_change": 1.1,
}


def _make_yolo_prediction(
    detections: list[tuple[int, float, float, float, float, float]],
):
    """Build a mock YOLO prediction result.

    Each detection is ``(class_id, cx, cy, w, h, confidence)``.
    Returns a list with one mock prediction object (like ``model.predict()``).
    """
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
    """Create *n* dummy Frame objects with image paths and timestamps."""
    frames = []
    for i in range(n):
        p = tmp_path / f"frame_{i:03d}.jpg"
        p.write_bytes(b"fake")
        frames.append(
            Frame(
                frame_id=f"frame_{i:03d}",
                image_path=p,
                timestamp=datetime(2024, 1, 1, 12, 0, i),
            )
        )
    return frames


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_direct_init(self) -> None:
        model = FsmTrackingModel(
            yolo_model=MagicMock(),
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,
        )
        assert isinstance(model, FsmTrackingModel)

    @patch("src.model.load_model_package")
    def test_from_package(self, mock_load: MagicMock, tmp_path: Path) -> None:
        pkg = MagicMock()
        pkg.model = MagicMock(name="FakeYOLO")
        pkg.infer_params = INFER_PARAMS
        pkg.prefilter_params = PREFILTER_PARAMS
        pkg.tracker_params = TRACKER_PARAMS
        mock_load.return_value = pkg

        archive_path = tmp_path / "model.zip"
        model = FsmTrackingModel.from_package(archive_path)

        mock_load.assert_called_once_with(archive_path)
        assert isinstance(model, FsmTrackingModel)


# ---------------------------------------------------------------------------
# Predict — basic scenarios
# ---------------------------------------------------------------------------


class TestPredictNoDetections:
    def test_returns_negative(self, tmp_path: Path) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([])

        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,
        )
        frames = _make_frames(5, tmp_path)
        output = model.predict(frames)

        assert isinstance(output, TemporalModelOutput)
        assert output.is_positive is False
        assert output.trigger_frame_index is None


class TestPredictConsecutiveHits:
    """YOLO returns the same detection in every frame -> tracker confirms."""

    def test_returns_positive(self, tmp_path: Path) -> None:
        det = (0, 0.5, 0.5, 0.1, 0.1, 0.8)  # class, cx, cy, w, h, conf
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([det])

        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,
        )
        # min_consecutive=3, so 5 frames is enough
        frames = _make_frames(5, tmp_path)
        output = model.predict(frames)

        assert output.is_positive is True
        assert output.trigger_frame_index is not None
        # Confirmed at index min_consecutive - 1 = 2 (0-indexed)
        assert output.trigger_frame_index == 2

    def test_trigger_frame_index_matches_min_consecutive(self, tmp_path: Path) -> None:
        det = (0, 0.5, 0.5, 0.1, 0.1, 0.8)
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([det])

        params = {**TRACKER_PARAMS, "min_consecutive": 4}
        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=params,
        )
        frames = _make_frames(6, tmp_path)
        output = model.predict(frames)

        assert output.is_positive is True
        assert output.trigger_frame_index == 3


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


class TestPadding:
    def test_short_sequence_gets_padded(self, tmp_path: Path) -> None:
        """A 2-frame sequence with min_consecutive=3 should still work."""
        det = (0, 0.5, 0.5, 0.1, 0.1, 0.8)
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([det])

        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,  # min_consecutive=3
        )
        # Only 2 frames — less than min_consecutive
        frames = _make_frames(2, tmp_path)
        output = model.predict(frames)

        # After padding to 3 frames, the same detection in all → confirmed
        assert output.is_positive is True


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestDetectionFiltering:
    def test_low_confidence_filtered(self, tmp_path: Path) -> None:
        """Detections below prefilter confidence are removed."""
        # conf=0.1 is below prefilter threshold of 0.3
        det = (0, 0.5, 0.5, 0.1, 0.1, 0.1)
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([det])

        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,
        )
        frames = _make_frames(5, tmp_path)
        output = model.predict(frames)

        assert output.is_positive is False

    def test_large_area_filtered(self, tmp_path: Path) -> None:
        """Detections larger than max_detection_area are removed."""
        # w=0.5, h=0.5 → area=0.25 > max_detection_area=0.05
        det = (0, 0.5, 0.5, 0.5, 0.5, 0.8)
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([det])

        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,
        )
        frames = _make_frames(5, tmp_path)
        output = model.predict(frames)

        assert output.is_positive is False


# ---------------------------------------------------------------------------
# Output details
# ---------------------------------------------------------------------------


class TestOutputDetails:
    def test_details_keys(self, tmp_path: Path) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([])

        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,
        )
        output = model.predict(_make_frames(5, tmp_path))

        assert "num_tracks" in output.details
        assert "num_confirmed_tracks" in output.details
        assert "num_detections_total" in output.details

    def test_positive_details(self, tmp_path: Path) -> None:
        det = (0, 0.5, 0.5, 0.1, 0.1, 0.8)
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([det])

        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,
        )
        output = model.predict(_make_frames(5, tmp_path))

        assert output.details["num_confirmed_tracks"] >= 1
        assert output.details["num_detections_total"] == 5


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------


class TestTimestampNone:
    def test_none_timestamp_handled(self, tmp_path: Path) -> None:
        """Frames with timestamp=None should not crash."""
        yolo = MagicMock()
        yolo.predict.return_value = _make_yolo_prediction([])

        model = FsmTrackingModel(
            yolo_model=yolo,
            infer_params=INFER_PARAMS,
            prefilter_params=PREFILTER_PARAMS,
            tracker_params=TRACKER_PARAMS,
        )
        frames = []
        for i in range(5):
            p = tmp_path / f"frame_{i:03d}.jpg"
            p.write_bytes(b"fake")
            frames.append(
                Frame(frame_id=f"frame_{i:03d}", image_path=p, timestamp=None)
            )

        output = model.predict(frames)
        assert output.is_positive is False

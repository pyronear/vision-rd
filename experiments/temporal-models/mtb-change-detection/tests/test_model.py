"""Integration tests for MtbChangeDetectionModel."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
from pyrocore import Frame

from mtb_change_detection.model import MtbChangeDetectionModel


def _make_yolo_result(detections: list[tuple[float, float, float, float, float]]):
    """Create a mock YOLO prediction result."""
    result = MagicMock()
    if not detections:
        result.boxes = MagicMock()
        result.boxes.__len__ = lambda self: 0
        result.boxes.__bool__ = lambda self: False
        return result

    n = len(detections)
    xywhn = torch.tensor([[d[0], d[1], d[2], d[3]] for d in detections])
    conf = torch.tensor([d[4] for d in detections])
    cls = torch.zeros(n)

    result.boxes = MagicMock()
    result.boxes.__len__ = lambda self: n
    result.boxes.__bool__ = lambda self: True
    result.boxes.xywhn = xywhn
    result.boxes.conf = conf
    result.boxes.cls = cls
    return result


class TestMtbChangeDetectionModel:
    @pytest.fixture()
    def model_params(self):
        return {
            "infer_params": {
                "confidence_threshold": 0.01,
                "iou_nms": 0.2,
                "image_size": 1024,
            },
            "prefilter_params": {
                "confidence_threshold": 0.01,
                "max_detection_area": 1.0,
            },
            "change_params": {
                "pixel_threshold": 19,
                "min_change_ratio": 0.01,
            },
            "tracker_params": {
                "iou_threshold": 0.1,
                "min_consecutive": 2,
                "max_misses": 0,
                "use_confidence_filter": False,
                "min_mean_confidence": 0.0,
                "use_area_change_filter": False,
                "min_area_change": 1.0,
            },
            "min_sequence_length": 2,
        }

    @pytest.fixture()
    def mock_yolo(self):
        return MagicMock()

    def _create_frames_with_images(
        self, tmp_path: Path, num_frames: int, images: list[np.ndarray]
    ):
        """Create Frame objects with actual image files."""
        frames = []
        for i in range(num_frames):
            ts = datetime(2024, 1, 1, 0, i, 0)
            frame_id = f"cam_2024-01-01T00-{i:02d}-00"
            img_path = tmp_path / f"{frame_id}.jpg"
            cv2.imwrite(str(img_path), images[i])
            frames.append(
                Frame(
                    frame_id=frame_id,
                    image_path=img_path,
                    timestamp=ts,
                )
            )
        return frames

    def test_no_detections_returns_negative(self, tmp_path, model_params, mock_yolo):
        mock_yolo.predict.return_value = [_make_yolo_result([])]

        model = MtbChangeDetectionModel(yolo_model=mock_yolo, **model_params)

        # Create identical frames (no change)
        img = np.full((100, 100), 128, dtype=np.uint8)
        frames = self._create_frames_with_images(tmp_path, 3, [img, img, img])

        output = model.predict(frames)
        assert output.is_positive is False
        assert output.trigger_frame_index is None

    def test_detections_with_change_triggers_alarm(
        self, tmp_path, model_params, mock_yolo
    ):
        """Detections in a changing region should trigger alarm."""
        # Detection at center of image
        det = (0.5, 0.5, 0.4, 0.4, 0.9)
        mock_yolo.predict.return_value = [_make_yolo_result([det])]

        model = MtbChangeDetectionModel(yolo_model=mock_yolo, **model_params)

        # Frame 0: dark, Frame 1+: bright in center (significant change)
        img0 = np.zeros((100, 100), dtype=np.uint8)
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img1[30:70, 30:70] = 100  # Big change in center
        img2 = np.zeros((100, 100), dtype=np.uint8)
        img2[30:70, 30:70] = 150  # More change

        frames = self._create_frames_with_images(tmp_path, 3, [img0, img1, img2])
        output = model.predict(frames)
        assert output.is_positive is True

    def test_detections_without_change_stays_negative(
        self, tmp_path, model_params, mock_yolo
    ):
        """Detections in a static region should NOT trigger alarm."""
        det = (0.5, 0.5, 0.4, 0.4, 0.9)
        mock_yolo.predict.return_value = [_make_yolo_result([det])]

        model = MtbChangeDetectionModel(yolo_model=mock_yolo, **model_params)

        # All frames identical — no change
        img = np.full((100, 100), 128, dtype=np.uint8)
        frames = self._create_frames_with_images(tmp_path, 4, [img] * 4)

        output = model.predict(frames)
        assert output.is_positive is False

    def test_change_outside_detection_bbox_stays_negative(
        self, tmp_path, model_params, mock_yolo
    ):
        """Change outside the detection bbox should not validate it."""
        # Detection in top-left corner
        det = (0.1, 0.1, 0.1, 0.1, 0.9)
        mock_yolo.predict.return_value = [_make_yolo_result([det])]

        model = MtbChangeDetectionModel(yolo_model=mock_yolo, **model_params)

        # Change only in bottom-right corner (outside detection bbox)
        img0 = np.zeros((100, 100), dtype=np.uint8)
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img1[80:100, 80:100] = 200

        frames = self._create_frames_with_images(tmp_path, 4, [img0, img1, img1, img1])
        output = model.predict(frames)
        assert output.is_positive is False

    def test_pixel_threshold_filters_small_changes(
        self, tmp_path, model_params, mock_yolo
    ):
        """Changes below pixel_threshold should be ignored."""
        det = (0.5, 0.5, 0.8, 0.8, 0.9)
        mock_yolo.predict.return_value = [_make_yolo_result([det])]

        # Set high pixel threshold
        model_params["change_params"]["pixel_threshold"] = 100
        model = MtbChangeDetectionModel(yolo_model=mock_yolo, **model_params)

        # Small change (diff = 50, below threshold 100)
        img0 = np.full((100, 100), 100, dtype=np.uint8)
        img1 = np.full((100, 100), 150, dtype=np.uint8)

        frames = self._create_frames_with_images(tmp_path, 4, [img0, img1, img1, img1])
        output = model.predict(frames)
        assert output.is_positive is False

    def test_output_details_contain_change_info(
        self, tmp_path, model_params, mock_yolo
    ):
        mock_yolo.predict.return_value = [_make_yolo_result([])]
        model = MtbChangeDetectionModel(yolo_model=mock_yolo, **model_params)

        img = np.full((100, 100), 128, dtype=np.uint8)
        frames = self._create_frames_with_images(tmp_path, 3, [img, img, img])

        output = model.predict(frames)
        assert "change_params" in output.details
        assert "num_detections_pre_change" in output.details
        assert "num_detections_total" in output.details

    def test_first_frame_detections_discarded(self, tmp_path, model_params, mock_yolo):
        """First frame has no previous frame, so detections should be discarded."""
        det = (0.5, 0.5, 0.4, 0.4, 0.9)
        mock_yolo.predict.return_value = [_make_yolo_result([det])]

        # Only 1 frame with min_consecutive=2 — should never alarm
        model_params["min_sequence_length"] = 1
        model_params["tracker_params"]["min_consecutive"] = 1
        model = MtbChangeDetectionModel(yolo_model=mock_yolo, **model_params)

        img = np.full((100, 100), 128, dtype=np.uint8)
        frames = self._create_frames_with_images(tmp_path, 1, [img])

        output = model.predict(frames)
        # Even with min_consecutive=1, first frame is discarded
        assert output.is_positive is False

    def test_padded_identical_frames_discard_detections(
        self, tmp_path, model_params, mock_yolo
    ):
        """Padded frames duplicate the same image, so change is zero."""
        det = (0.5, 0.5, 0.4, 0.4, 0.9)
        mock_yolo.predict.return_value = [_make_yolo_result([det])]

        # Pad a single frame to min_sequence_length=4
        model_params["min_sequence_length"] = 4
        model_params["tracker_params"]["min_consecutive"] = 2
        model = MtbChangeDetectionModel(yolo_model=mock_yolo, **model_params)

        # Single image — padding will duplicate it
        img = np.full((100, 100), 128, dtype=np.uint8)
        frames = self._create_frames_with_images(tmp_path, 1, [img])

        output = model.predict(frames)
        # All padded frames compare identical images → zero change → no alarm
        assert output.is_positive is False
        assert output.details["num_detections_total"] == 0

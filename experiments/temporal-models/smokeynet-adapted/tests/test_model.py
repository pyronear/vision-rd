"""Tests for smokeynet_adapted.model.

YOLO model and image I/O are mocked to avoid GPU/disk dependencies in CI.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from pyrocore.types import Frame

from smokeynet_adapted.model import SmokeyNetModel
from smokeynet_adapted.net import SmokeyNetAdapted


def _make_model(d_model=32):
    """Create a SmokeyNetModel with fake YOLO and small net."""
    yolo_model = MagicMock()
    net = SmokeyNetAdapted(
        d_model=d_model,
        lstm_layers=1,
        spatial_layers=1,
        spatial_heads=4,
        dropout=0.0,
    )

    # Create a fake RoI extractor
    roi_extractor = MagicMock()

    config = {
        "infer": {
            "confidence_threshold": 0.01,
            "iou_nms": 0.2,
            "image_size": 1024,
        },
        "extract": {
            "roi_size": 7,
            "context_factor": 1.2,
            "max_detections_per_frame": 10,
        },
        "tubes": {"iou_threshold": 0.2, "max_misses": 2},
        "train": {"d_model": d_model},
        "classification_threshold": 0.5,
    }

    return SmokeyNetModel(
        yolo_model=yolo_model,
        net=net,
        roi_extractor=roi_extractor,
        config=config,
    )


def _make_frames(n=3):
    return [
        Frame(
            frame_id=f"frame_{i:03d}",
            image_path=Path(f"/fake/frame_{i:03d}.jpg"),
            timestamp=None,
        )
        for i in range(n)
    ]


class TestSmokeyNetModelConstruction:
    def test_direct_init(self):
        model = _make_model()
        assert model._net is not None

    def test_config_accessible(self):
        model = _make_model()
        assert model._config["infer"]["image_size"] == 1024


class TestPredictNoDetections:
    def test_returns_negative(self):
        model = _make_model(d_model=32)
        frames = _make_frames(3)

        # Mock YOLO to return no detections
        model._yolo_model.predict.return_value = [MagicMock(boxes=None)]

        # Mock _extract_roi_features to return empty tensors
        empty_feats = torch.zeros(0, 32)
        empty_idx = torch.zeros(0, dtype=torch.long)
        empty_bbox = torch.zeros(0, 4)

        with patch.object(
            model,
            "_extract_roi_features",
            return_value=(empty_feats, empty_idx, empty_bbox),
        ):
            output = model.predict(frames)

        assert output.is_positive is False
        assert output.trigger_frame_index is None


class TestPredictPositive:
    def test_returns_output_with_details(self):
        model = _make_model(d_model=32)
        frames = _make_frames(3)

        # Mock YOLO to return detections
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.xywhn = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        mock_boxes.conf = torch.tensor([0.9])
        mock_boxes.cls = torch.tensor([0], dtype=torch.int32)
        mock_boxes.__len__ = lambda self: 1
        mock_result.boxes = mock_boxes
        model._yolo_model.predict.return_value = [mock_result]

        # Mock RoI features
        feats = torch.randn(3, 32)
        idx = torch.tensor([0, 1, 2])
        bbox = torch.randn(3, 4)

        with patch.object(
            model,
            "_extract_roi_features",
            return_value=(feats, idx, bbox),
        ):
            output = model.predict(frames)

        assert "probability" in output.details
        assert "num_tubes" in output.details
        assert "num_detections_total" in output.details


class TestOutputDetails:
    def test_details_keys(self):
        model = _make_model(d_model=32)
        frames = _make_frames(2)

        model._yolo_model.predict.return_value = [MagicMock(boxes=None)]

        empty_feats = torch.zeros(0, 32)
        empty_idx = torch.zeros(0, dtype=torch.long)
        empty_bbox = torch.zeros(0, 4)

        with patch.object(
            model,
            "_extract_roi_features",
            return_value=(empty_feats, empty_idx, empty_bbox),
        ):
            output = model.predict(frames)

        expected_keys = {
            "probability",
            "num_tubes",
            "num_detections_total",
            "num_frames",
        }
        assert set(output.details.keys()) == expected_keys

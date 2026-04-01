"""Tests for PyroDetectorModel."""

import json
from unittest.mock import MagicMock, patch

from PIL import Image as PILImage
from pyrocore import Frame, TemporalModel

from pyro_detector_baseline.model import PyroDetectorModel


def _make_frames(tmp_path, n=5):
    """Create Frame objects with valid 1x1 JPEG image files."""
    frames = []
    for i in range(n):
        img = tmp_path / f"cam_2023-01-01T00-00-{i:02d}.jpg"
        PILImage.new("RGB", (1, 1)).save(img)
        frames.append(
            Frame(
                frame_id=img.stem,
                image_path=img,
                timestamp=None,
            )
        )
    return frames


@patch("pyro_detector_baseline.predictor_wrapper.pyro_predictor")
class TestPyroDetectorModel:
    def _make_model(self, mock_pyro, predict_values=None, conf_thresh=0.35):
        """Create a PyroDetectorModel with mocked Predictor."""
        mock_predictor = MagicMock()
        mock_predictor.conf_thresh = conf_thresh
        if predict_values is not None:
            mock_predictor.predict.side_effect = list(predict_values)
        else:
            mock_predictor.predict.return_value = 0.0
        mock_pyro.Predictor.return_value = mock_predictor

        model = PyroDetectorModel(conf_thresh=conf_thresh)
        return model

    def test_is_temporal_model_subclass(self, mock_pyro):
        model = self._make_model(mock_pyro)
        assert isinstance(model, TemporalModel)

    def test_predict_no_detections(self, mock_pyro, tmp_path):
        model = self._make_model(mock_pyro)
        frames = _make_frames(tmp_path, n=5)

        output = model.predict(frames)

        assert output.is_positive is False
        assert output.trigger_frame_index is None
        assert output.details["max_confidence"] == 0.0

    def test_predict_above_threshold(self, mock_pyro, tmp_path):
        model = self._make_model(mock_pyro, predict_values=[0.0, 0.0, 0.6, 0.7, 0.8])
        frames = _make_frames(tmp_path, n=5)

        output = model.predict(frames)

        assert output.is_positive is True
        assert output.trigger_frame_index == 2

    def test_output_details_keys(self, mock_pyro, tmp_path):
        model = self._make_model(mock_pyro)
        frames = _make_frames(tmp_path, n=3)

        output = model.predict(frames)

        expected_keys = {
            "per_frame_confidences",
            "conf_thresh",
            "num_frames",
            "num_detections_total",
            "max_confidence",
        }
        assert set(output.details.keys()) == expected_keys

    def test_details_json_serializable(self, mock_pyro, tmp_path):
        model = self._make_model(mock_pyro, predict_values=[0.1, 0.2, 0.3])
        frames = _make_frames(tmp_path, n=3)

        output = model.predict(frames)

        # Should not raise
        json.dumps(output.details)

    def test_unique_cam_ids_across_calls(self, mock_pyro, tmp_path):
        model = self._make_model(mock_pyro)
        frames = _make_frames(tmp_path, n=2)

        # Call predict twice
        model.predict(frames)
        model.predict(frames)

        # Extract cam_ids from all predict calls
        mock_predictor = mock_pyro.Predictor.return_value
        cam_ids = set()
        for call in mock_predictor.predict.call_args_list:
            cam_id = call.kwargs.get("cam_id") or call[1].get("cam_id")
            cam_ids.add(cam_id)

        # Should have 2 different cam_ids
        assert len(cam_ids) == 2

    def test_num_detections_total(self, mock_pyro, tmp_path):
        model = self._make_model(mock_pyro, predict_values=[0.0, 0.1, 0.0, 0.2, 0.0])
        frames = _make_frames(tmp_path, n=5)

        output = model.predict(frames)

        # 2 frames had confidence > 0
        assert output.details["num_detections_total"] == 2

    def test_empty_frames(self, mock_pyro, tmp_path):
        model = self._make_model(mock_pyro)

        output = model.predict([])

        assert output.is_positive is False
        assert output.trigger_frame_index is None
        assert output.details["num_frames"] == 0
        assert output.details["max_confidence"] == 0.0

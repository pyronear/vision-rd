"""Tests for predictor_wrapper module."""

from unittest.mock import MagicMock

from PIL import Image

from pyro_detector_baseline.predictor_wrapper import predict_sequence


def _make_frame_paths(tmp_path, n=5):
    """Create valid 1x1 JPEG image files and return their paths."""
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        p = images / f"cam_2023-01-01T00-00-{i:02d}.jpg"
        Image.new("RGB", (1, 1)).save(p)
        paths.append(p)
    return paths


def _mock_predictor(conf_thresh=0.35, predict_values=None):
    """Create a mock Predictor with conf_thresh attribute."""
    predictor = MagicMock()
    predictor.conf_thresh = conf_thresh
    if predict_values is not None:
        predictor.predict.side_effect = list(predict_values)
    else:
        predictor.predict.return_value = 0.0
    return predictor


class TestPredictSequence:
    def test_all_zeros_returns_negative(self, tmp_path):
        paths = _make_frame_paths(tmp_path, n=5)
        predictor = _mock_predictor()

        is_alarm, trigger_idx, confs = predict_sequence(predictor, paths, cam_id="test")

        assert is_alarm is False
        assert trigger_idx is None
        assert confs == [0.0] * 5
        assert predictor.predict.call_count == 5

    def test_above_threshold_returns_positive(self, tmp_path):
        paths = _make_frame_paths(tmp_path, n=5)
        predictor = _mock_predictor(
            conf_thresh=0.35, predict_values=[0.0, 0.0, 0.6, 0.7, 0.8]
        )

        is_alarm, trigger_idx, confs = predict_sequence(predictor, paths, cam_id="test")

        assert is_alarm is True
        assert trigger_idx == 2
        assert confs == [0.0, 0.0, 0.6, 0.7, 0.8]

    def test_trigger_at_first_frame(self, tmp_path):
        paths = _make_frame_paths(tmp_path, n=3)
        predictor = _mock_predictor(conf_thresh=0.35, predict_values=[0.9, 0.8, 0.7])

        is_alarm, trigger_idx, confs = predict_sequence(predictor, paths, cam_id="test")

        assert is_alarm is True
        assert trigger_idx == 0

    def test_empty_frames(self, tmp_path):
        predictor = _mock_predictor()

        is_alarm, trigger_idx, confs = predict_sequence(predictor, [], cam_id="test")

        assert is_alarm is False
        assert trigger_idx is None
        assert confs == []
        predictor.predict.assert_not_called()

    def test_cam_id_passed_to_predictor(self, tmp_path):
        paths = _make_frame_paths(tmp_path, n=2)
        predictor = _mock_predictor()

        predict_sequence(predictor, paths, cam_id="my_seq")

        for call in predictor.predict.call_args_list:
            assert (
                call.kwargs.get("cam_id") == "my_seq"
                or call[1].get("cam_id") == "my_seq"
            )

    def test_confidence_just_below_threshold(self, tmp_path):
        paths = _make_frame_paths(tmp_path, n=3)
        predictor = _mock_predictor(conf_thresh=0.35, predict_values=[0.34, 0.34, 0.35])

        is_alarm, trigger_idx, confs = predict_sequence(predictor, paths, cam_id="test")

        # 0.35 is not > 0.35 (strict inequality matches production)
        assert is_alarm is False
        assert trigger_idx is None

    def test_confidence_above_threshold(self, tmp_path):
        paths = _make_frame_paths(tmp_path, n=3)
        predictor = _mock_predictor(conf_thresh=0.35, predict_values=[0.0, 0.36, 0.6])

        is_alarm, trigger_idx, confs = predict_sequence(predictor, paths, cam_id="test")

        assert is_alarm is True
        assert trigger_idx == 1

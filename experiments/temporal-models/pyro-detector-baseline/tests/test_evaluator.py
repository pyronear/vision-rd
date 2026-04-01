"""Tests for evaluator module."""

from pyro_detector_baseline.evaluator import (
    compute_metrics,
    compute_single_frame_baseline,
)


def _result(gt, pred, n_det=0, first_ts=None, conf_ts=None):
    """Build a minimal sequence result dict."""
    return {
        "is_positive_gt": gt,
        "is_positive_pred": pred,
        "num_detections_total": n_det,
        "confirmed_timestamp": conf_ts,
        "first_timestamp": first_ts,
    }


class TestComputeMetrics:
    def test_perfect_predictions(self):
        results = [
            _result(True, True),
            _result(False, False),
        ]
        m = compute_metrics(results)
        assert m["tp"] == 1
        assert m["tn"] == 1
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["fpr"] == 0.0

    def test_all_false_positives(self):
        results = [
            _result(False, True),
            _result(False, True),
        ]
        m = compute_metrics(results)
        assert m["fp"] == 2
        assert m["precision"] == 0.0
        assert m["fpr"] == 1.0

    def test_all_false_negatives(self):
        results = [
            _result(True, False),
            _result(True, False),
        ]
        m = compute_metrics(results)
        assert m["fn"] == 2
        assert m["recall"] == 0.0

    def test_mixed(self):
        results = [
            _result(True, True),
            _result(True, False),
            _result(False, True),
            _result(False, False),
        ]
        m = compute_metrics(results)
        assert m["tp"] == 1
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["tn"] == 1
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5

    def test_empty(self):
        m = compute_metrics([])
        assert m["num_sequences"] == 0
        assert m["precision"] == 0.0

    def test_ttd_computed(self):
        results = [
            _result(
                True,
                True,
                first_ts="2024-01-01T12:00:00",
                conf_ts="2024-01-01T12:01:00",
            ),
            _result(
                True,
                True,
                first_ts="2024-01-01T12:00:00",
                conf_ts="2024-01-01T12:02:00",
            ),
        ]
        m = compute_metrics(results)
        assert m["mean_ttd_seconds"] == 90.0
        assert m["median_ttd_seconds"] == 90.0

    def test_ttd_none_when_no_tp(self):
        results = [_result(True, False)]
        m = compute_metrics(results)
        assert m["mean_ttd_seconds"] is None
        assert m["median_ttd_seconds"] is None


class TestComputeSingleFrameBaseline:
    def test_any_detection_triggers_alarm(self):
        results = [
            _result(True, False, n_det=3, first_ts="2024-01-01T12:00:00"),
            _result(False, False, n_det=1, first_ts="2024-01-01T12:00:00"),
        ]
        m = compute_single_frame_baseline(results)
        # Both have detections, so both predicted positive
        assert m["tp"] == 1
        assert m["fp"] == 1

    def test_no_detections_all_negative(self):
        results = [
            _result(True, False, n_det=0, first_ts="2024-01-01T12:00:00"),
            _result(False, False, n_det=0, first_ts="2024-01-01T12:00:00"),
        ]
        m = compute_single_frame_baseline(results)
        assert m["fn"] == 1
        assert m["tn"] == 1
        assert m["tp"] == 0
        assert m["fp"] == 0

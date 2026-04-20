"""Tests for temporal_model_leaderboard.metrics."""

from temporal_model_leaderboard.metrics import compute_metrics
from temporal_model_leaderboard.types import SequenceResult


def _make_result(gt: bool, pred: bool, ttd: int | None = None) -> SequenceResult:
    return SequenceResult(
        sequence_id="test",
        ground_truth=gt,
        predicted=pred,
        ttd_frames=ttd,
    )


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=1),
            _make_result(gt=True, pred=True, ttd=3),
            _make_result(gt=False, pred=False),
            _make_result(gt=False, pred=False),
        ]
        m = compute_metrics("test-model", results)

        assert m.tp == 2
        assert m.fp == 0
        assert m.fn == 0
        assert m.tn == 2
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.fpr == 0.0
        assert m.mean_ttd_frames == 2.0
        assert m.median_ttd_frames == 2.0

    def test_all_false_positives(self) -> None:
        results = [
            _make_result(gt=False, pred=True),
            _make_result(gt=False, pred=True),
        ]
        m = compute_metrics("test", results)

        assert m.tp == 0
        assert m.fp == 2
        assert m.precision == 0.0
        assert m.fpr == 1.0

    def test_all_false_negatives(self) -> None:
        results = [
            _make_result(gt=True, pred=False),
            _make_result(gt=True, pred=False),
        ]
        m = compute_metrics("test", results)

        assert m.fn == 2
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_mixed(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=4),
            _make_result(gt=True, pred=False),
            _make_result(gt=False, pred=True),
            _make_result(gt=False, pred=False),
        ]
        m = compute_metrics("test", results)

        assert m.tp == 1
        assert m.fp == 1
        assert m.fn == 1
        assert m.tn == 1
        assert m.precision == 0.5
        assert m.recall == 0.5
        assert m.mean_ttd_frames == 4.0

    def test_empty_results(self) -> None:
        m = compute_metrics("test", [])

        assert m.num_sequences == 0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.mean_ttd_frames is None
        assert m.median_ttd_frames is None

    def test_ttd_none_excluded(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=None),
            _make_result(gt=True, pred=True, ttd=2),
        ]
        m = compute_metrics("test", results)

        assert m.mean_ttd_frames == 2.0
        assert m.median_ttd_frames == 2.0

    def test_ttd_all_none(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=None),
        ]
        m = compute_metrics("test", results)

        assert m.mean_ttd_frames is None
        assert m.median_ttd_frames is None

    def test_median_even_count(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=1),
            _make_result(gt=True, pred=True, ttd=2),
            _make_result(gt=True, pred=True, ttd=3),
            _make_result(gt=True, pred=True, ttd=4),
        ]
        m = compute_metrics("test", results)

        assert m.median_ttd_frames == 2.5

    def test_model_name_preserved(self) -> None:
        m = compute_metrics("my-model", [])
        assert m.model_name == "my-model"

    def test_num_sequences(self) -> None:
        results = [_make_result(gt=True, pred=True, ttd=0)] * 5
        m = compute_metrics("test", results)
        assert m.num_sequences == 5

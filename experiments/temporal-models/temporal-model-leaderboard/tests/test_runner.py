"""Tests for temporal_model_leaderboard.runner."""

from pathlib import Path

from pyrocore import Frame, TemporalModel, TemporalModelOutput

from temporal_model_leaderboard.runner import evaluate_model


class AlwaysPositiveModel(TemporalModel):
    """Dummy model that always predicts positive, triggering at frame 2."""

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        idx = min(2, len(frames) - 1) if frames else None
        return TemporalModelOutput(is_positive=True, trigger_frame_index=idx)


class AlwaysNegativeModel(TemporalModel):
    """Dummy model that always predicts negative."""

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        return TemporalModelOutput(is_positive=False)


def _create_test_dataset(base: Path) -> Path:
    """Create a minimal test dataset with one WF and one FP sequence."""
    test_dir = base / "test"

    # Wildfire sequence with 3 frames
    wf_seq = test_dir / "wildfire" / "wf_site_001_2024-01-01T10-00-00"
    (wf_seq / "images").mkdir(parents=True)
    (wf_seq / "labels").mkdir(parents=True)
    for ts in [
        "2024-01-01T10-00-00",
        "2024-01-01T10-00-30",
        "2024-01-01T10-01-00",
    ]:
        (wf_seq / "images" / f"wf_site_001_{ts}.jpg").touch()

    # FP sequence with 3 frames
    fp_seq = test_dir / "fp" / "fp_site_002_2024-01-01T12-00-00"
    (fp_seq / "images").mkdir(parents=True)
    (fp_seq / "labels").mkdir(parents=True)
    for ts in [
        "2024-01-01T12-00-00",
        "2024-01-01T12-00-30",
        "2024-01-01T12-01-00",
    ]:
        (fp_seq / "images" / f"fp_site_002_{ts}.jpg").touch()

    return test_dir


class TestEvaluateModel:
    def test_always_positive_model(self, tmp_path: Path) -> None:
        test_dir = _create_test_dataset(tmp_path)
        model = AlwaysPositiveModel()

        results = evaluate_model(model, test_dir)

        assert len(results) == 2

        wf_result = next(r for r in results if r.ground_truth)
        assert wf_result.predicted is True
        assert wf_result.ttd_frames == 2  # trigger at frame 2

        fp_result = next(r for r in results if not r.ground_truth)
        assert fp_result.predicted is True
        assert fp_result.ttd_frames is None  # not a TP

    def test_always_negative_model(self, tmp_path: Path) -> None:
        test_dir = _create_test_dataset(tmp_path)
        model = AlwaysNegativeModel()

        results = evaluate_model(model, test_dir)

        assert len(results) == 2
        assert all(not r.predicted for r in results)
        assert all(r.ttd_frames is None for r in results)

    def test_skips_empty_sequences(self, tmp_path: Path) -> None:
        test_dir = _create_test_dataset(tmp_path)
        # Add a sequence with no images
        empty_seq = test_dir / "wildfire" / "empty_seq"
        empty_seq.mkdir(parents=True)

        model = AlwaysNegativeModel()
        results = evaluate_model(model, test_dir)

        ids = {r.sequence_id for r in results}
        assert "empty_seq" not in ids

    def test_sequence_ids(self, tmp_path: Path) -> None:
        test_dir = _create_test_dataset(tmp_path)
        model = AlwaysNegativeModel()

        results = evaluate_model(model, test_dir)

        ids = {r.sequence_id for r in results}
        assert "wf_site_001_2024-01-01T10-00-00" in ids
        assert "fp_site_002_2024-01-01T12-00-00" in ids


class TestTTDComputation:
    def test_ttd_trigger_at_first_frame(self, tmp_path: Path) -> None:
        """TTD should be 0 when trigger is at the first frame."""
        test_dir = _create_test_dataset(tmp_path)

        class TriggerAtZero(TemporalModel):
            def predict(self, frames: list[Frame]) -> TemporalModelOutput:
                return TemporalModelOutput(is_positive=True, trigger_frame_index=0)

        model = TriggerAtZero()
        results = evaluate_model(model, test_dir)

        wf_result = next(r for r in results if r.ground_truth)
        assert wf_result.ttd_frames == 0

    def test_ttd_none_when_no_trigger_index(self, tmp_path: Path) -> None:
        """TTD should be None when trigger_frame_index is None."""
        test_dir = _create_test_dataset(tmp_path)

        class PositiveNoIndex(TemporalModel):
            def predict(self, frames: list[Frame]) -> TemporalModelOutput:
                return TemporalModelOutput(is_positive=True, trigger_frame_index=None)

        model = PositiveNoIndex()
        results = evaluate_model(model, test_dir)

        wf_result = next(r for r in results if r.ground_truth)
        assert wf_result.ttd_frames is None

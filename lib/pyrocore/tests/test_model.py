"""Tests for the TemporalModel ABC and supporting helpers."""

from datetime import datetime
from pathlib import Path

from pyrocore.model import TemporalModel, _try_parse_timestamp
from pyrocore.types import Frame, TemporalModelOutput

# ---------------------------------------------------------------------------
# _try_parse_timestamp
# ---------------------------------------------------------------------------


class TestTryParseTimestamp:
    def test_pyronear_convention(self):
        result = _try_parse_timestamp("adf_site_999_2023-05-23T17-18-31")
        assert result == datetime(2023, 5, 23, 17, 18, 31)

    def test_simple_timestamp(self):
        result = _try_parse_timestamp("2024-01-15T09-30-00")
        assert result == datetime(2024, 1, 15, 9, 30, 0)

    def test_no_timestamp_returns_none(self):
        assert _try_parse_timestamp("random_filename") is None

    def test_empty_string_returns_none(self):
        assert _try_parse_timestamp("") is None

    def test_partial_timestamp_returns_none(self):
        assert _try_parse_timestamp("prefix_2023-05-23") is None


# ---------------------------------------------------------------------------
# TemporalModel — load_sequence
# ---------------------------------------------------------------------------


class TestLoadSequence:
    def _make_model(self):
        """Create a minimal concrete subclass for testing."""

        class DummyModel(TemporalModel):
            def predict(self, frames: list[Frame]) -> TemporalModelOutput:
                return TemporalModelOutput(is_positive=False)

        return DummyModel()

    def test_default_load_sequence(self, tmp_path):
        img = tmp_path / "adf_site_999_2023-05-23T17-18-31.jpg"
        img.touch()

        model = self._make_model()
        frames = model.load_sequence([img])

        assert len(frames) == 1
        assert frames[0].frame_id == "adf_site_999_2023-05-23T17-18-31"
        assert frames[0].image_path == img
        assert frames[0].timestamp == datetime(2023, 5, 23, 17, 18, 31)

    def test_unparseable_timestamp_defaults_to_none(self, tmp_path):
        img = tmp_path / "some_image.png"
        img.touch()

        model = self._make_model()
        frames = model.load_sequence([img])

        assert frames[0].timestamp is None

    def test_preserves_order(self, tmp_path):
        paths = []
        for name in ["c.jpg", "a.jpg", "b.jpg"]:
            p = tmp_path / name
            p.touch()
            paths.append(p)

        model = self._make_model()
        frames = model.load_sequence(paths)

        assert [f.frame_id for f in frames] == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# TemporalModel — predict_sequence (template method)
# ---------------------------------------------------------------------------


class TestPredictSequence:
    def test_wires_load_and_predict(self, tmp_path):
        img = tmp_path / "frame_2024-01-01T00-00-00.jpg"
        img.touch()

        class AlwaysPositive(TemporalModel):
            def predict(self, frames: list[Frame]) -> TemporalModelOutput:
                return TemporalModelOutput(
                    is_positive=True,
                    trigger_frame_index=0,
                    details={"num_frames": len(frames)},
                )

        model = AlwaysPositive()
        output = model.predict_sequence([img])

        assert output.is_positive is True
        assert output.trigger_frame_index == 0
        assert output.details == {"num_frames": 1}

    def test_custom_load_sequence(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.touch()

        class CustomLoader(TemporalModel):
            def load_sequence(self, frames: list[Path]) -> list[Frame]:
                return [
                    Frame(
                        frame_id="custom_id",
                        image_path=p,
                        timestamp=datetime(2000, 1, 1),
                    )
                    for p in frames
                ]

            def predict(self, frames: list[Frame]) -> TemporalModelOutput:
                assert frames[0].frame_id == "custom_id"
                assert frames[0].timestamp == datetime(2000, 1, 1)
                return TemporalModelOutput(is_positive=False)

        model = CustomLoader()
        output = model.predict_sequence([img])

        assert output.is_positive is False

    def test_negative_prediction(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.touch()

        class AlwaysNegative(TemporalModel):
            def predict(self, frames: list[Frame]) -> TemporalModelOutput:
                return TemporalModelOutput(is_positive=False)

        model = AlwaysNegative()
        output = model.predict_sequence([img])

        assert output.is_positive is False
        assert output.trigger_frame_index is None
        assert output.details == {}

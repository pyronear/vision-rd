from datetime import datetime

import pytest

from src.data import is_wf_sequence, pad_sequence, parse_timestamp
from src.types import Detection, FrameResult


class TestParseTimestamp:
    def test_standard_filename(self):
        ts = parse_timestamp("cam1_2024-06-15T14-30-00.jpg")
        assert ts == datetime(2024, 6, 15, 14, 30, 0)

    def test_stem_only(self):
        ts = parse_timestamp("prefix_2024-01-01T00-00-00")
        assert ts == datetime(2024, 1, 1, 0, 0, 0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            parse_timestamp("no_timestamp_here.jpg")


class TestIsWfSequence:
    def test_five_columns_is_wf(self, tmp_path):
        seq = tmp_path / "seq1"
        labels = seq / "labels"
        labels.mkdir(parents=True)
        (labels / "frame1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        assert is_wf_sequence(seq) is True

    def test_six_columns_is_fp(self, tmp_path):
        seq = tmp_path / "seq2"
        labels = seq / "labels"
        labels.mkdir(parents=True)
        (labels / "frame1.txt").write_text("0 0.5 0.5 0.2 0.2 0.85\n")
        assert is_wf_sequence(seq) is False

    def test_empty_labels_default_to_fp(self, tmp_path):
        seq = tmp_path / "seq3"
        labels = seq / "labels"
        labels.mkdir(parents=True)
        (labels / "frame1.txt").write_text("")
        assert is_wf_sequence(seq) is False

    def test_no_labels_dir_returns_false(self, tmp_path):
        seq = tmp_path / "seq4"
        seq.mkdir()
        assert is_wf_sequence(seq) is False

    def test_non_txt_files_ignored(self, tmp_path):
        seq = tmp_path / "seq5"
        labels = seq / "labels"
        labels.mkdir(parents=True)
        (labels / "frame1.xml").write_text("0 0.5 0.5 0.2 0.2")
        assert is_wf_sequence(seq) is False

    def test_mixed_empty_and_wf(self, tmp_path):
        seq = tmp_path / "seq6"
        labels = seq / "labels"
        labels.mkdir(parents=True)
        (labels / "frame1.txt").write_text("")
        (labels / "frame2.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        assert is_wf_sequence(seq) is True


class TestPadSequence:
    """Tests for pad_sequence (moved from test_tracker.py)."""

    def _det(self, cx=0.5, cy=0.5, w=0.2, h=0.2) -> Detection:
        return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9)

    def _frame(self, frame_id: str) -> FrameResult:
        return FrameResult(
            frame_id=frame_id,
            timestamp=datetime(2024, 1, 1),
            detections=[self._det()],
        )

    def test_already_long_enough(self):
        frames = [self._frame("f1"), self._frame("f2"), self._frame("f3")]
        result = pad_sequence(frames, 3)
        assert len(result) == 3

    def test_returns_new_list(self):
        frames = [self._frame("f1")]
        result = pad_sequence(frames, 1)
        assert result is not frames

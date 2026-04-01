from datetime import datetime

import pytest

from mtb_change_detection.data import (
    is_wf_sequence,
    list_sequences,
    pad_sequence,
    parse_timestamp,
)
from mtb_change_detection.types import Detection, FrameResult


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
    def test_wildfire_parent_is_wf(self, tmp_path):
        seq = tmp_path / "wildfire" / "seq1"
        seq.mkdir(parents=True)
        assert is_wf_sequence(seq) is True

    def test_fp_parent_is_not_wf(self, tmp_path):
        seq = tmp_path / "fp" / "seq2"
        seq.mkdir(parents=True)
        assert is_wf_sequence(seq) is False

    def test_other_parent_is_not_wf(self, tmp_path):
        seq = tmp_path / "other" / "seq3"
        seq.mkdir(parents=True)
        assert is_wf_sequence(seq) is False


class TestListSequences:
    def test_finds_sequences_in_nested_dirs(self, tmp_path):
        (tmp_path / "wildfire" / "seq_a").mkdir(parents=True)
        (tmp_path / "wildfire" / "seq_b").mkdir(parents=True)
        (tmp_path / "fp" / "seq_c").mkdir(parents=True)
        seqs = list_sequences(tmp_path)
        names = [s.name for s in seqs]
        assert names == ["seq_a", "seq_b", "seq_c"]

    def test_empty_split_dir(self, tmp_path):
        assert list_sequences(tmp_path) == []

    def test_only_wildfire(self, tmp_path):
        (tmp_path / "wildfire" / "seq1").mkdir(parents=True)
        seqs = list_sequences(tmp_path)
        assert len(seqs) == 1
        assert seqs[0].name == "seq1"

    def test_ignores_files(self, tmp_path):
        (tmp_path / "wildfire").mkdir()
        (tmp_path / "wildfire" / "readme.txt").write_text("hi")
        (tmp_path / "wildfire" / "seq1").mkdir()
        seqs = list_sequences(tmp_path)
        assert len(seqs) == 1

    def test_sorted_by_name(self, tmp_path):
        for name in ("z_seq", "a_seq", "m_seq"):
            (tmp_path / "fp" / name).mkdir(parents=True)
        seqs = list_sequences(tmp_path)
        names = [s.name for s in seqs]
        assert names == ["a_seq", "m_seq", "z_seq"]


class TestPadSequence:
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

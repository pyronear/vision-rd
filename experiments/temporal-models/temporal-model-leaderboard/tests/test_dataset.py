"""Tests for temporal_model_leaderboard.dataset."""

from pathlib import Path

import pytest

from temporal_model_leaderboard.dataset import (
    get_sorted_frames,
    list_sequences,
)


def _make_sequence(base: Path, category: str, name: str) -> Path:
    """Create a minimal sequence directory with images and labels."""
    seq = base / category / name
    (seq / "images").mkdir(parents=True)
    (seq / "labels").mkdir(parents=True)
    return seq


def _touch_image(seq: Path, timestamp: str) -> Path:
    """Create an empty .jpg file with the given timestamp suffix."""
    img = seq / "images" / f"cam_site_001_{timestamp}.jpg"
    img.touch()
    return img


class TestListSequences:
    def test_finds_wildfire_and_fp(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "test"
        _make_sequence(test_dir, "wildfire", "seq_a")
        _make_sequence(test_dir, "fp", "seq_b")

        result = list_sequences(test_dir)

        assert len(result) == 2
        wf = [(p.name, gt) for p, gt in result if gt]
        fp = [(p.name, gt) for p, gt in result if not gt]
        assert wf == [("seq_a", True)]
        assert fp == [("seq_b", False)]

    def test_empty_dir(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        assert list_sequences(test_dir) == []

    def test_missing_wildfire_dir(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "test"
        _make_sequence(test_dir, "fp", "seq_b")

        result = list_sequences(test_dir)
        assert len(result) == 1
        assert result[0][1] is False

    def test_missing_fp_dir(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "test"
        _make_sequence(test_dir, "wildfire", "seq_a")

        result = list_sequences(test_dir)
        assert len(result) == 1
        assert result[0][1] is True

    def test_sorted_within_category(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "test"
        _make_sequence(test_dir, "wildfire", "z_seq")
        _make_sequence(test_dir, "wildfire", "a_seq")

        result = list_sequences(test_dir)
        names = [p.name for p, _ in result]
        assert names == ["a_seq", "z_seq"]

    def test_non_directory_entries_ignored(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "test"
        _make_sequence(test_dir, "wildfire", "seq_a")
        (test_dir / "wildfire" / "stray_file.txt").touch()

        result = list_sequences(test_dir)
        assert len(result) == 1


class TestGetSortedFrames:
    def test_sorted_by_timestamp(self, tmp_path: Path) -> None:
        seq = _make_sequence(tmp_path, "wildfire", "seq1")
        _touch_image(seq, "2024-01-01T10-00-30")
        _touch_image(seq, "2024-01-01T10-00-00")
        _touch_image(seq, "2024-01-01T10-01-00")

        result = get_sorted_frames(seq)
        timestamps = [p.stem.split("_")[-1] for p in result]
        assert timestamps == [
            "2024-01-01T10-00-00",
            "2024-01-01T10-00-30",
            "2024-01-01T10-01-00",
        ]

    def test_no_images_dir(self, tmp_path: Path) -> None:
        seq = tmp_path / "seq_no_images"
        seq.mkdir()
        assert get_sorted_frames(seq) == []

    def test_empty_images_dir(self, tmp_path: Path) -> None:
        seq = _make_sequence(tmp_path, "wildfire", "seq_empty")
        assert get_sorted_frames(seq) == []

    def test_invalid_timestamp_raises(self, tmp_path: Path) -> None:
        seq = _make_sequence(tmp_path, "wildfire", "seq_bad")
        (seq / "images" / "no_timestamp.jpg").touch()

        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            get_sorted_frames(seq)

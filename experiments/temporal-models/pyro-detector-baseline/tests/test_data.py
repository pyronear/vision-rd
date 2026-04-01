"""Tests for data utilities."""

from datetime import datetime

import pytest

from pyro_detector_baseline.data import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
    parse_timestamp,
)

# --- parse_timestamp ---


class TestParseTimestamp:
    def test_standard_filename(self):
        ts = parse_timestamp("adf_site_999_2023-05-23T17-18-31.jpg")
        assert ts == datetime(2023, 5, 23, 17, 18, 31)

    def test_stem_only(self):
        ts = parse_timestamp("adf_site_999_2023-05-23T17-18-31")
        assert ts == datetime(2023, 5, 23, 17, 18, 31)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            parse_timestamp("no_timestamp_here.jpg")


# --- is_wf_sequence ---


class TestIsWfSequence:
    def test_five_column_is_wf(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "frame.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        assert is_wf_sequence(tmp_path) is True

    def test_six_column_is_fp(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "frame.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n")
        assert is_wf_sequence(tmp_path) is False

    def test_empty_labels_default_fp(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "frame.txt").write_text("")
        assert is_wf_sequence(tmp_path) is False

    def test_no_labels_dir(self, tmp_path):
        assert is_wf_sequence(tmp_path) is False

    def test_non_txt_files_ignored(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "frame.csv").write_text("0 0.5 0.5 0.1 0.1\n")
        assert is_wf_sequence(tmp_path) is False

    def test_parent_wildfire_is_positive(self, tmp_path):
        wf_dir = tmp_path / "wildfire" / "seq_a"
        wf_dir.mkdir(parents=True)
        assert is_wf_sequence(wf_dir) is True

    def test_parent_fp_is_negative(self, tmp_path):
        fp_dir = tmp_path / "fp" / "seq_a"
        fp_dir.mkdir(parents=True)
        assert is_wf_sequence(fp_dir) is False


# --- list_sequences ---


class TestListSequences:
    def test_lists_directories_sorted(self, tmp_path):
        (tmp_path / "seq_b").mkdir()
        (tmp_path / "seq_a").mkdir()
        (tmp_path / "file.txt").write_text("not a dir")
        result = list_sequences(tmp_path)
        assert result == [tmp_path / "seq_a", tmp_path / "seq_b"]

    def test_empty_directory(self, tmp_path):
        assert list_sequences(tmp_path) == []

    def test_nested_layout(self, tmp_path):
        """Nested pyro-dataset layout: split/{wildfire,fp}/seq/."""
        wf = tmp_path / "wildfire"
        fp = tmp_path / "fp"
        wf.mkdir()
        fp.mkdir()
        (wf / "seq_wf_a").mkdir()
        (wf / "seq_wf_b").mkdir()
        (fp / "seq_fp_c").mkdir()
        result = list_sequences(tmp_path)
        names = [p.name for p in result]
        assert names == ["seq_fp_c", "seq_wf_a", "seq_wf_b"]

    def test_nested_layout_wildfire_only(self, tmp_path):
        wf = tmp_path / "wildfire"
        wf.mkdir()
        (wf / "seq_a").mkdir()
        result = list_sequences(tmp_path)
        assert result == [wf / "seq_a"]


# --- get_sorted_frames ---


class TestGetSortedFrames:
    def test_returns_sorted_by_timestamp(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        (images / "cam_2023-01-01T00-00-30.jpg").write_text("")
        (images / "cam_2023-01-01T00-00-00.jpg").write_text("")
        (images / "cam_2023-01-01T00-01-00.jpg").write_text("")
        result = get_sorted_frames(tmp_path)
        assert [p.name for p in result] == [
            "cam_2023-01-01T00-00-00.jpg",
            "cam_2023-01-01T00-00-30.jpg",
            "cam_2023-01-01T00-01-00.jpg",
        ]

    def test_no_images_dir(self, tmp_path):
        assert get_sorted_frames(tmp_path) == []

    def test_empty_images_dir(self, tmp_path):
        (tmp_path / "images").mkdir()
        assert get_sorted_frames(tmp_path) == []

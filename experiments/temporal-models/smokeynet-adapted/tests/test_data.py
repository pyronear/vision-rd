"""Tests for label-file parsing."""

from pathlib import Path

import pytest

from smokeynet_adapted.data import load_detections


def _write_label(tmp_path: Path, frame_id: str, text: str) -> Path:
    seq = tmp_path / "seq_a"
    (seq / "labels").mkdir(parents=True, exist_ok=True)
    (seq / "labels" / f"{frame_id}.txt").write_text(text)
    return seq


def test_load_detections_5col_sets_confidence_to_one(tmp_path):
    seq = _write_label(tmp_path, "f1", "0 0.5 0.4 0.1 0.2\n")
    dets = load_detections(seq, "f1")
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 0
    assert d.cx == pytest.approx(0.5)
    assert d.cy == pytest.approx(0.4)
    assert d.w == pytest.approx(0.1)
    assert d.h == pytest.approx(0.2)
    assert d.confidence == pytest.approx(1.0)


def test_load_detections_6col_reads_confidence_from_last_column(tmp_path):
    seq = _write_label(tmp_path, "f1", "0 0.25 0.30 0.05 0.07 0.42\n")
    dets = load_detections(seq, "f1")
    assert len(dets) == 1
    assert dets[0].confidence == pytest.approx(0.42)


def test_load_detections_empty_file_returns_empty_list(tmp_path):
    seq = _write_label(tmp_path, "f1", "")
    assert load_detections(seq, "f1") == []


def test_load_detections_missing_file_returns_empty_list(tmp_path):
    seq = _write_label(tmp_path, "f1", "")
    assert load_detections(seq, "nonexistent") == []

"""Tests for fire-tube construction."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.tube import (
    build_tubes_for_sequence,
    compute_iou,
    crop_detection,
    match_detections,
)
from src.types import Detection, FrameResult

# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


class TestComputeIou:
    def test_identical(self):
        d = Detection(0, 0.5, 0.5, 0.2, 0.2, 0.9)
        assert compute_iou(d, d) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = Detection(0, 0.1, 0.1, 0.1, 0.1, 0.9)
        b = Detection(0, 0.9, 0.9, 0.1, 0.1, 0.9)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = Detection(0, 0.5, 0.5, 0.2, 0.2, 0.9)
        b = Detection(0, 0.55, 0.55, 0.2, 0.2, 0.9)
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        a = Detection(0, 0.5, 0.5, 0.0, 0.0, 0.9)
        b = Detection(0, 0.5, 0.5, 0.2, 0.2, 0.9)
        assert compute_iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# Match detections
# ---------------------------------------------------------------------------


class TestMatchDetections:
    def test_empty_prev(self):
        curr = [Detection(0, 0.5, 0.5, 0.2, 0.2, 0.9)]
        assert match_detections([], curr, 0.1) == []

    def test_empty_curr(self):
        prev = [Detection(0, 0.5, 0.5, 0.2, 0.2, 0.9)]
        assert match_detections(prev, [], 0.1) == []

    def test_one_to_one_match(self):
        d = Detection(0, 0.5, 0.5, 0.2, 0.2, 0.9)
        matches = match_detections([d], [d], 0.1)
        assert len(matches) == 1
        assert matches[0][0] == 0
        assert matches[0][1] == 0

    def test_no_match_below_threshold(self):
        a = Detection(0, 0.1, 0.1, 0.05, 0.05, 0.9)
        b = Detection(0, 0.9, 0.9, 0.05, 0.05, 0.9)
        assert match_detections([a], [b], 0.1) == []


# ---------------------------------------------------------------------------
# Crop detection
# ---------------------------------------------------------------------------


class TestCropDetection:
    def test_basic_crop(self, tmp_path: Path):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        det = Detection(0, 0.5, 0.5, 0.2, 0.2, 0.9)
        crop = crop_detection(img_path, det, crop_size=32)

        assert crop.shape == (32, 32, 3)
        assert crop.dtype == np.uint8

    def test_crop_at_boundary(self, tmp_path: Path):
        img = Image.new("RGB", (100, 100), (0, 255, 0))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        det = Detection(0, 0.0, 0.0, 0.2, 0.2, 0.9)
        crop = crop_detection(img_path, det, crop_size=16)

        assert crop.shape == (16, 16, 3)

    def test_degenerate_box(self, tmp_path: Path):
        img = Image.new("RGB", (100, 100), (0, 0, 255))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        det = Detection(0, 0.5, 0.5, 0.0, 0.0, 0.9)
        crop = crop_detection(img_path, det, crop_size=16)

        assert crop.shape == (16, 16, 3)
        assert crop.sum() == 0  # zero-filled


# ---------------------------------------------------------------------------
# Build tubes
# ---------------------------------------------------------------------------


class TestBuildTubes:
    def _make_frame(self, frame_id: str, detections: list[Detection]) -> FrameResult:
        return FrameResult(
            frame_id=frame_id,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            detections=detections,
        )

    def _make_images(self, tmp_path: Path, frame_ids: list[str]) -> Path:
        """Create dummy images in tmp_path."""
        for fid in frame_ids:
            img = Image.new("RGB", (100, 100), (128, 128, 128))
            img.save(tmp_path / f"{fid}.jpg")
        return tmp_path

    def test_no_detections(self, tmp_path: Path):
        frames = [self._make_frame("f1", []), self._make_frame("f2", [])]
        image_dir = self._make_images(tmp_path, ["f1", "f2"])

        tubes = build_tubes_for_sequence(
            frames, image_dir, "seq1", confidence_threshold=0.1
        )
        assert len(tubes) == 0

    def test_single_detection_single_frame(self, tmp_path: Path):
        det = Detection(0, 0.5, 0.5, 0.1, 0.1, 0.8)
        frames = [self._make_frame("f1", [det])]
        image_dir = self._make_images(tmp_path, ["f1"])

        tubes = build_tubes_for_sequence(
            frames, image_dir, "seq1", confidence_threshold=0.1
        )
        assert len(tubes) == 1
        assert len(tubes[0].crops) == 1

    def test_tracked_detection_across_frames(self, tmp_path: Path):
        det = Detection(0, 0.5, 0.5, 0.1, 0.1, 0.8)
        frames = [
            self._make_frame("f1", [det]),
            self._make_frame("f2", [det]),
            self._make_frame("f3", [det]),
        ]
        image_dir = self._make_images(tmp_path, ["f1", "f2", "f3"])

        tubes = build_tubes_for_sequence(
            frames, image_dir, "seq1", confidence_threshold=0.1
        )
        assert len(tubes) == 1
        assert len(tubes[0].crops) == 3

    def test_confidence_filter(self, tmp_path: Path):
        det = Detection(0, 0.5, 0.5, 0.1, 0.1, 0.1)  # low confidence
        frames = [self._make_frame("f1", [det])]
        image_dir = self._make_images(tmp_path, ["f1"])

        tubes = build_tubes_for_sequence(
            frames, image_dir, "seq1", confidence_threshold=0.3
        )
        assert len(tubes) == 0

    def test_area_filter(self, tmp_path: Path):
        det = Detection(0, 0.5, 0.5, 0.5, 0.5, 0.8)  # area = 0.25
        frames = [self._make_frame("f1", [det])]
        image_dir = self._make_images(tmp_path, ["f1"])

        tubes = build_tubes_for_sequence(
            frames,
            image_dir,
            "seq1",
            confidence_threshold=0.1,
            max_detection_area=0.05,
        )
        assert len(tubes) == 0

"""Unit tests for the MTB change detection algorithm."""

import numpy as np
import pytest

from mtb_change_detection.change_detector import (
    compute_change_mask,
    compute_change_ratio_in_bbox,
)


class TestComputeChangeMask:
    def test_identical_frames_produce_no_change(self):
        frame = np.full((100, 100), 128, dtype=np.uint8)
        mask = compute_change_mask(frame, frame, threshold=19)
        assert not mask.any()

    def test_completely_different_frames(self):
        frame_a = np.zeros((100, 100), dtype=np.uint8)
        frame_b = np.full((100, 100), 255, dtype=np.uint8)
        mask = compute_change_mask(frame_a, frame_b, threshold=19)
        assert mask.all()

    def test_single_pixel_above_threshold(self):
        frame_a = np.zeros((10, 10), dtype=np.uint8)
        frame_b = np.zeros((10, 10), dtype=np.uint8)
        frame_b[5, 5] = 20  # diff = 20 > threshold 19
        mask = compute_change_mask(frame_a, frame_b, threshold=19)
        assert mask[5, 5]
        assert mask.sum() == 1

    def test_single_pixel_at_threshold_not_detected(self):
        frame_a = np.zeros((10, 10), dtype=np.uint8)
        frame_b = np.zeros((10, 10), dtype=np.uint8)
        frame_b[5, 5] = 19  # diff == threshold, should NOT pass (strict >)
        mask = compute_change_mask(frame_a, frame_b, threshold=19)
        assert not mask[5, 5]

    def test_below_threshold_not_detected(self):
        frame_a = np.zeros((10, 10), dtype=np.uint8)
        frame_b = np.full((10, 10), 10, dtype=np.uint8)
        mask = compute_change_mask(frame_a, frame_b, threshold=19)
        assert not mask.any()

    def test_non_square_image(self):
        frame_a = np.zeros((50, 200), dtype=np.uint8)
        frame_b = np.full((50, 200), 100, dtype=np.uint8)
        mask = compute_change_mask(frame_a, frame_b, threshold=19)
        assert mask.shape == (50, 200)
        assert mask.all()

    def test_reverse_order_same_result(self):
        frame_a = np.zeros((10, 10), dtype=np.uint8)
        frame_b = np.full((10, 10), 50, dtype=np.uint8)
        mask_ab = compute_change_mask(frame_a, frame_b, threshold=19)
        mask_ba = compute_change_mask(frame_b, frame_a, threshold=19)
        np.testing.assert_array_equal(mask_ab, mask_ba)


class TestComputeChangeRatioInBbox:
    def test_all_changed_returns_one(self):
        mask = np.ones((100, 100), dtype=bool)
        ratio = compute_change_ratio_in_bbox(mask, cx=0.5, cy=0.5, w=0.5, h=0.5)
        assert ratio == pytest.approx(1.0)

    def test_no_change_returns_zero(self):
        mask = np.zeros((100, 100), dtype=bool)
        ratio = compute_change_ratio_in_bbox(mask, cx=0.5, cy=0.5, w=0.5, h=0.5)
        assert ratio == 0.0

    def test_half_changed(self):
        mask = np.zeros((100, 100), dtype=bool)
        # Top half of image is changed
        mask[:50, :] = True
        # Bbox covers full image
        ratio = compute_change_ratio_in_bbox(mask, cx=0.5, cy=0.5, w=1.0, h=1.0)
        assert ratio == pytest.approx(0.5, abs=0.01)

    def test_zero_area_bbox_returns_zero(self):
        mask = np.ones((100, 100), dtype=bool)
        ratio = compute_change_ratio_in_bbox(mask, cx=0.5, cy=0.5, w=0.0, h=0.0)
        assert ratio == 0.0

    def test_bbox_at_edge(self):
        mask = np.ones((100, 100), dtype=bool)
        # Bbox at bottom-right corner
        ratio = compute_change_ratio_in_bbox(mask, cx=0.9, cy=0.9, w=0.2, h=0.2)
        assert ratio == pytest.approx(1.0)

    def test_bbox_outside_image_clamped(self):
        mask = np.ones((100, 100), dtype=bool)
        # Bbox extends beyond image
        ratio = compute_change_ratio_in_bbox(mask, cx=1.0, cy=1.0, w=0.5, h=0.5)
        # Should still return valid ratio for the clamped region
        assert 0.0 <= ratio <= 1.0

    def test_localized_change_inside_bbox(self):
        mask = np.zeros((100, 100), dtype=bool)
        # Change only in center 10x10 area
        mask[45:55, 45:55] = True
        # Bbox covers center region
        ratio = compute_change_ratio_in_bbox(mask, cx=0.5, cy=0.5, w=0.2, h=0.2)
        assert ratio > 0.0

    def test_localized_change_outside_bbox(self):
        mask = np.zeros((100, 100), dtype=bool)
        # Change in top-left corner
        mask[:10, :10] = True
        # Bbox in bottom-right
        ratio = compute_change_ratio_in_bbox(mask, cx=0.8, cy=0.8, w=0.2, h=0.2)
        assert ratio == 0.0

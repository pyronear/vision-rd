"""Tests for threshold calibration."""

import numpy as np
import pytest

from smokeynet_adapted.calibration import calibrate_threshold


class TestCalibrateThreshold:
    def test_picks_smallest_threshold_achieving_recall(self) -> None:
        # 4 positives at probs [0.9, 0.8, 0.7, 0.2]; 4 negatives at [0.1..0.4].
        probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.15, 0.3, 0.4])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        # target_recall = 0.75 -> need 3 of 4 positives. Threshold just below 0.7.
        t = calibrate_threshold(probs, labels, target_recall=0.75)
        assert 0.4 < t <= 0.7

    def test_recall_1_requires_lowest_positive(self) -> None:
        probs = np.array([0.9, 0.2, 0.1])
        labels = np.array([1, 1, 0])
        t = calibrate_threshold(probs, labels, target_recall=1.0)
        assert t <= 0.2

    def test_raises_if_no_positives(self) -> None:
        probs = np.array([0.1, 0.2])
        labels = np.array([0, 0])
        with pytest.raises(ValueError, match="no positives"):
            calibrate_threshold(probs, labels, target_recall=0.95)

    def test_raises_if_unreachable_recall(self) -> None:
        probs = np.array([0.5, 0.5])
        labels = np.array([1, 0])
        with pytest.raises(ValueError):
            calibrate_threshold(probs, labels, target_recall=1.5)

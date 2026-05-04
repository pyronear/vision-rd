"""Tests for tabular feature extraction."""

from datetime import datetime

import numpy as np
import pytest

from src.features import TABULAR_FEATURE_DIM, extract_tabular_features
from src.types import Detection, FireTube, TubeCrop


def _make_crop(
    cx: float = 0.5,
    cy: float = 0.5,
    w: float = 0.1,
    h: float = 0.1,
    conf: float = 0.8,
    intensity: int = 128,
) -> TubeCrop:
    image = np.full((64, 64, 3), intensity, dtype=np.uint8)
    return TubeCrop(
        frame_id="f1",
        timestamp=datetime(2024, 1, 1),
        image=image,
        detection=Detection(0, cx, cy, w, h, conf),
    )


class TestExtractTabularFeatures:
    def test_output_dimension(self):
        tube = FireTube(
            tube_id=0,
            sequence_id="seq1",
            crops=[_make_crop(), _make_crop()],
        )
        features = extract_tabular_features(tube)
        assert features.shape == (TABULAR_FEATURE_DIM,)

    def test_single_crop_tube(self):
        tube = FireTube(
            tube_id=0,
            sequence_id="seq1",
            crops=[_make_crop()],
        )
        features = extract_tabular_features(tube)
        assert features.shape == (TABULAR_FEATURE_DIM,)
        # Pair features should be zero, global features populated
        assert features[0:20].sum() == 0.0  # pair aggregates are zero
        assert features[20] == 1.0  # tube_length = 1

    def test_empty_tube(self):
        tube = FireTube(tube_id=0, sequence_id="seq1", crops=[])
        features = extract_tabular_features(tube)
        assert features.shape == (TABULAR_FEATURE_DIM,)
        assert features.sum() == 0.0

    def test_identical_crops_minimal_change(self):
        crop = _make_crop()
        tube = FireTube(
            tube_id=0,
            sequence_id="seq1",
            crops=[crop, crop, crop],
        )
        features = extract_tabular_features(tube)
        assert features.shape == (TABULAR_FEATURE_DIM,)
        # Area ratio should be ~1.0 (mean of pair area ratios)
        assert features[0] == pytest.approx(1.0, abs=0.01)
        # Centroid shift should be ~0
        assert features[1] == pytest.approx(0.0, abs=0.01)

    def test_growing_detection_area(self):
        crop1 = _make_crop(w=0.1, h=0.1)
        crop2 = _make_crop(w=0.2, h=0.2)
        tube = FireTube(
            tube_id=0,
            sequence_id="seq1",
            crops=[crop1, crop2],
        )
        features = extract_tabular_features(tube)
        # Total area change (global feature at index 21)
        # area_first = 0.01, area_last = 0.04 -> ratio = 4.0
        assert features[21] == pytest.approx(4.0, abs=0.01)

    def test_different_intensities(self):
        crop1 = _make_crop(intensity=50)
        crop2 = _make_crop(intensity=200)
        tube = FireTube(
            tube_id=0,
            sequence_id="seq1",
            crops=[crop1, crop2],
        )
        features = extract_tabular_features(tube)
        # Intensity change should be non-zero (pair feature index 2)
        assert features[2] > 0

    def test_mean_confidence(self):
        crop1 = _make_crop(conf=0.6)
        crop2 = _make_crop(conf=0.8)
        tube = FireTube(
            tube_id=0,
            sequence_id="seq1",
            crops=[crop1, crop2],
        )
        features = extract_tabular_features(tube)
        # Mean confidence (global feature at index 23)
        assert features[23] == pytest.approx(0.7, abs=0.01)

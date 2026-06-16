"""Temporal feature extraction from fire-tubes.

Extracts tabular features that capture how a candidate detection evolves
over time.  Designed for Pyronear's 30-second frame interval where optical
flow is not meaningful.

Feature vector (24 dimensions per tube):
- Per consecutive pair (5 raw features): area change ratio, centroid shift,
  mean intensity change, intensity histogram distance, confidence.
- Aggregated across pairs: mean, std, min, max -> 5 * 4 = 20 dims.
- Global features: tube length, total area change, total centroid drift,
  mean confidence -> 4 dims.
"""

import math

import numpy as np

from src.types import FireTube

TABULAR_FEATURE_DIM = 24


def extract_tabular_features(tube: FireTube) -> np.ndarray:
    """Extract a 24-dimensional tabular feature vector from a fire-tube.

    For tubes with a single crop, pair features are zero-filled and only
    global features are populated.

    Returns:
        1-D numpy array of shape ``(24,)``.
    """
    crops = tube.crops
    n = len(crops)

    if n == 0:
        return np.zeros(TABULAR_FEATURE_DIM, dtype=np.float64)

    # --- Compute per-pair features ---
    pair_features = []

    for i in range(n - 1):
        c_prev = crops[i]
        c_curr = crops[i + 1]
        d_prev = c_prev.detection
        d_curr = c_curr.detection

        # Area change ratio
        area_prev = d_prev.w * d_prev.h
        area_curr = d_curr.w * d_curr.h
        area_ratio = area_curr / area_prev if area_prev > 0 else 0.0

        # Centroid shift (Euclidean distance in normalized coords)
        centroid_shift = math.sqrt(
            (d_curr.cx - d_prev.cx) ** 2 + (d_curr.cy - d_prev.cy) ** 2
        )

        # Mean intensity change
        mean_prev = c_prev.image.mean()
        mean_curr = c_curr.image.mean()
        intensity_change = abs(float(mean_curr) - float(mean_prev))

        # Intensity histogram distance (chi-squared on 8-bin grayscale)
        hist_prev = _grayscale_histogram(c_prev.image)
        hist_curr = _grayscale_histogram(c_curr.image)
        hist_dist = _chi_squared_distance(hist_prev, hist_curr)

        # Confidence of current frame
        confidence = d_curr.confidence

        pair_features.append(
            [area_ratio, centroid_shift, intensity_change, hist_dist, confidence]
        )

    # --- Aggregate pair features ---
    if pair_features:
        pairs_arr = np.array(pair_features)  # shape (n-1, 5)
        agg_mean = pairs_arr.mean(axis=0)  # 5 dims
        agg_std = pairs_arr.std(axis=0)  # 5 dims
        agg_min = pairs_arr.min(axis=0)  # 5 dims
        agg_max = pairs_arr.max(axis=0)  # 5 dims
        pair_agg = np.concatenate([agg_mean, agg_std, agg_min, agg_max])  # 20 dims
    else:
        pair_agg = np.zeros(20, dtype=np.float64)

    # --- Global features ---
    tube_length = float(n)

    d_first = crops[0].detection
    d_last = crops[-1].detection
    area_first = d_first.w * d_first.h
    area_last = d_last.w * d_last.h
    total_area_change = area_last / area_first if area_first > 0 else 0.0

    total_centroid_drift = math.sqrt(
        (d_last.cx - d_first.cx) ** 2 + (d_last.cy - d_first.cy) ** 2
    )

    mean_confidence = np.mean([c.detection.confidence for c in crops])

    global_feats = np.array(
        [tube_length, total_area_change, total_centroid_drift, mean_confidence]
    )

    return np.concatenate([pair_agg, global_feats])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grayscale_histogram(image: np.ndarray, bins: int = 8) -> np.ndarray:
    """Compute a normalized grayscale histogram from an RGB image."""
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    return hist


def _chi_squared_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Compute chi-squared distance between two histograms."""
    denom = hist_a + hist_b
    # Avoid division by zero
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.sum((hist_a[mask] - hist_b[mask]) ** 2 / denom[mask]))

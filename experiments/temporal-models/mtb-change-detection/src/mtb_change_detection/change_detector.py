"""Core MTB change detection algorithm.

Implements pixel-wise frame differencing and per-detection change
validation, inspired by the SlowFastMTB paper (Choi, Kim & Oh, 2022).

All functions are pure (no I/O, no state) and operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np


def compute_change_mask(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    threshold: int = 19,
) -> np.ndarray:
    """Pixel-wise absolute difference thresholded to a binary change mask.

    Args:
        frame_a: Grayscale uint8 image (H, W).
        frame_b: Grayscale uint8 image (H, W), same shape as *frame_a*.
        threshold: Pixel intensity difference cutoff.  Differences strictly
            greater than this value are marked as changed.

    Returns:
        Boolean 2-D array (H, W) where ``True`` indicates change.
    """
    diff = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
    return diff > threshold


def compute_change_ratio_in_bbox(
    change_mask: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> float:
    """Fraction of changed pixels within a normalized bounding box.

    Args:
        change_mask: Boolean 2-D array (H, W).
        cx: Normalized center-x of the bounding box.
        cy: Normalized center-y of the bounding box.
        w: Normalized width of the bounding box.
        h: Normalized height of the bounding box.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 if the box has zero area.
    """
    img_h, img_w = change_mask.shape

    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w, int((cx + w / 2) * img_w))
    y2 = min(img_h, int((cy + h / 2) * img_h))

    region = change_mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(region.sum()) / region.size

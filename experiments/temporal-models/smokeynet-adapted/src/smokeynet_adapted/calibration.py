"""Decision threshold calibration from val predictions.

Picks the smallest sigmoid probability threshold that achieves at least the
requested recall on the supplied (probs, labels) pairs. Used by the packager
to pin ``decision.threshold`` before building the archive.
"""

import numpy as np


def calibrate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    target_recall: float,
) -> float:
    """Return the smallest ``p`` s.t. recall at threshold ``p`` >= ``target_recall``.

    Args:
        probs: 1D array of sigmoid probabilities (one per val sample).
        labels: 1D array of 0/1 ground truth, same length as ``probs``.
        target_recall: Desired recall on the positive class; must be in ``(0, 1]``.

    Returns:
        Threshold in ``[0, 1]``.

    Raises:
        ValueError: if ``labels`` has no positives, if arrays are mis-shaped,
            or if ``target_recall`` is not in ``(0, 1]``.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall!r}")
    if probs.shape != labels.shape or probs.ndim != 1:
        raise ValueError("probs and labels must be equal-length 1D arrays")

    pos_probs = np.sort(probs[labels == 1])
    if pos_probs.size == 0:
        raise ValueError("no positives in labels; cannot calibrate recall")

    # We want recall = (#pos with prob >= t) / n_pos >= target_recall.
    # Equivalently: at most floor(n_pos * (1 - target_recall)) positives may
    # fall below t. Sorting pos_probs ascending, the threshold is the
    # (n_drop)-th element; returning it means we include it in the "at or
    # above threshold" set (recall counts prob >= t).
    n_pos = pos_probs.size
    n_drop = int(np.floor(n_pos * (1.0 - target_recall)))
    return float(pos_probs[n_drop])

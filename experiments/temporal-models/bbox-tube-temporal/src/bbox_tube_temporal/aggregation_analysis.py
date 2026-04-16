"""Offline analysis of sequence-level aggregation rules over per-tube logits.

Reads predictions.json files produced by scripts/evaluate_packaged.py and
derives sequence-level scores under alternative aggregation rules (max,
top-k-mean). Supports threshold sweeps, target-recall search, and
confusion-matrix derivation.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

AGGREGATION_RULES = ("max", "top_k_mean")


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Load per-sequence prediction records, sorted by sequence_id.

    Accepts the JSON written by scripts/evaluate_packaged.py, which uses
    json.dump with default settings (so +/-Infinity serializes as the
    non-strict "Infinity" / "-Infinity" literals that json.loads handles).
    """
    records = json.loads(predictions_path.read_text())
    records.sort(key=lambda r: r["sequence_id"])
    return records


def aggregate_score(tube_logits: list[float], *, rule: str, k: int) -> float:
    """Aggregate per-tube logits into a single sequence-level score.

    Rules:
        * ``max``: maximum logit across all tubes. ``k`` is ignored.
        * ``top_k_mean``: mean of the k largest logits. If fewer than k
          tubes exist, returns ``-inf`` (sequence cannot clear the rule).

    Empty tube list always returns ``-inf``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if rule not in AGGREGATION_RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {AGGREGATION_RULES}")
    if not tube_logits:
        return -np.inf
    arr = np.asarray(tube_logits, dtype=float)
    if rule == "max":
        return float(arr.max())
    # top_k_mean
    if arr.size < k:
        return -np.inf
    top_k = np.partition(arr, -k)[-k:]
    return float(top_k.mean())

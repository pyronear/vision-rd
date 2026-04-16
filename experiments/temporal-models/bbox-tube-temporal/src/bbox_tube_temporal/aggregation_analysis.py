"""Offline analysis of sequence-level aggregation rules over per-tube logits.

Reads predictions.json files produced by scripts/evaluate_packaged.py and
derives sequence-level scores under alternative aggregation rules (max,
top-k-mean). Supports threshold sweeps, target-recall search, and
confusion-matrix derivation.
"""

import json
from pathlib import Path
from typing import Any


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Load per-sequence prediction records, sorted by sequence_id.

    Accepts the JSON written by scripts/evaluate_packaged.py, which uses
    json.dump with default settings (so +/-Infinity serializes as the
    non-strict "Infinity" / "-Infinity" literals that json.loads handles).
    """
    records = json.loads(predictions_path.read_text())
    records.sort(key=lambda r: r["sequence_id"])
    return records

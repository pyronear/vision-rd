"""Turn per-sequence predictions + folder ground truth into FP/FN review sets.

Outputs (per (model, split)):
  - ``fp_sequences.csv``       — one row per FP, ordered by sequence name.
  - ``fn_sequences.csv``       — one row per FN, ordered by sequence name.
  - ``summary.json``           — counts + confusion matrix.
  - ``review_manifest.json``   — machine-readable input for build_fiftyone.

Usage::

    uv run python scripts/build_review_sets.py \\
        --model-name <model-name> \\
        --split <split> \\
        --split-dir data/01_raw/datasets/<split> \\
        --predictions-path data/07_model_output/<model-name>/<split>/predictions.json \\
        --output-dir data/08_reporting/<model-name>/<split>
"""

import argparse
import csv
import dataclasses
import json
import logging
from pathlib import Path

from data_quality_sequential.dataset import SequenceRef, iter_sequences
from data_quality_sequential.review import (
    Prediction,
    ReviewSet,
    build_review_sets,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "sequence_name",
    "split",
    "model_name",
    "ground_truth",
    "predicted",
    "trigger_frame_index",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _load_predictions(path: Path) -> list[Prediction]:
    raw = json.loads(path.read_text())
    return [Prediction(**entry) for entry in raw]


def _write_csv(review_set: ReviewSet, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for entry in review_set.entries:
            writer.writerow(dataclasses.asdict(entry))


def _confusion_matrix(
    refs: list[SequenceRef], preds_by_name: dict[str, Prediction]
) -> dict[str, int]:
    tp = fp = fn = tn = missing = 0
    for ref in refs:
        pred = preds_by_name.get(ref.name)
        if pred is None:
            missing += 1
            continue
        if ref.ground_truth and pred.predicted:
            tp += 1
        elif not ref.ground_truth and pred.predicted:
            fp += 1
        elif ref.ground_truth and not pred.predicted:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "missing": missing}


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    refs = list(iter_sequences(args.split_dir, split=args.split))
    predictions = _load_predictions(args.predictions_path)
    logger.info(
        "Loaded %d refs and %d predictions for %s/%s",
        len(refs),
        len(predictions),
        args.model_name,
        args.split,
    )

    fp_set, fn_set = build_review_sets(
        refs, predictions, split=args.split, model_name=args.model_name
    )

    _write_csv(fp_set, args.output_dir / "fp_sequences.csv")
    _write_csv(fn_set, args.output_dir / "fn_sequences.csv")

    summary = {
        "model_name": args.model_name,
        "split": args.split,
        "num_refs": len(refs),
        "num_predictions": len(predictions),
        "confusion_matrix": _confusion_matrix(
            refs, {p.sequence_name: p for p in predictions}
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    manifest = {
        "model_name": args.model_name,
        "split": args.split,
        "split_dir": str(args.split_dir),
        "fp": [dataclasses.asdict(e) for e in fp_set.entries],
        "fn": [dataclasses.asdict(e) for e in fn_set.entries],
    }
    (args.output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    logger.info(
        "Wrote FP=%d FN=%d to %s",
        len(fp_set.entries),
        len(fn_set.entries),
        args.output_dir,
    )


if __name__ == "__main__":
    main()

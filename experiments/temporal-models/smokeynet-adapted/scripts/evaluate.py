"""Evaluate SmokeyNetAdapted on a data split.

Loads a packaged model, runs it on every sequence in the split, and
writes per-sequence results and aggregated metrics.

Usage:
    uv run python scripts/evaluate.py \
        --data-dir data/01_raw/datasets/val \
        --model-package data/06_models/model.zip \
        --output-dir data/08_reporting/val
"""

import argparse
import json
import logging
import statistics
from pathlib import Path

from pyrocore.types import Frame
from tqdm import tqdm

from smokeynet_adapted.data import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
    parse_timestamp,
)
from smokeynet_adapted.model import SmokeyNetModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SmokeyNetAdapted on a data split."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.model_package)
    model = SmokeyNetModel.from_package(args.model_package)

    sequences = list_sequences(args.data_dir)
    logger.info("Found %d sequences in %s", len(sequences), args.data_dir)

    results = []
    for seq_dir in tqdm(sequences, desc="Evaluating"):
        seq_id = seq_dir.name
        gt = is_wf_sequence(seq_dir)

        frame_paths = get_sorted_frames(seq_dir)
        if not frame_paths:
            continue

        frames = [
            Frame(
                frame_id=p.stem,
                image_path=p,
                timestamp=parse_timestamp(p.stem),
            )
            for p in frame_paths
        ]

        output = model.predict(frames)

        # Compute TTD for true positives
        ttd_seconds = None
        if gt and output.is_positive and output.trigger_frame_index is not None:
            first_ts = frames[0].timestamp
            trigger_ts = frames[output.trigger_frame_index].timestamp
            if first_ts is not None and trigger_ts is not None:
                ttd_seconds = (trigger_ts - first_ts).total_seconds()

        results.append(
            {
                "sequence_id": seq_id,
                "ground_truth": gt,
                "predicted": output.is_positive,
                "probability": output.details.get("probability"),
                "ttd_seconds": ttd_seconds,
                "num_detections": output.details.get("num_detections_total", 0),
                "num_tubes": output.details.get("num_tubes", 0),
            }
        )

    # Compute metrics
    tp = sum(1 for r in results if r["ground_truth"] and r["predicted"])
    fp = sum(1 for r in results if not r["ground_truth"] and r["predicted"])
    fn = sum(1 for r in results if r["ground_truth"] and not r["predicted"])
    tn = sum(1 for r in results if not r["ground_truth"] and not r["predicted"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    ttd_values = [r["ttd_seconds"] for r in results if r["ttd_seconds"] is not None]
    mean_ttd = statistics.mean(ttd_values) if ttd_values else None
    median_ttd = statistics.median(ttd_values) if ttd_values else None

    metrics = {
        "num_sequences": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "mean_ttd_seconds": (round(mean_ttd, 1) if mean_ttd is not None else None),
        "median_ttd_seconds": (
            round(median_ttd, 1) if median_ttd is not None else None
        ),
    }

    # Save results
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("Saved %d results to %s", len(results), results_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info(
        "  P=%.4f  R=%.4f  F1=%.4f  FPR=%.4f",
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["fpr"],
    )
    if mean_ttd is not None:
        logger.info(
            "  Mean TTD=%.1fs  Median TTD=%.1fs",
            mean_ttd,
            median_ttd,
        )
    logger.info("  TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)


if __name__ == "__main__":
    main()

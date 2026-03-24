"""Sweep tracking parameters on cached inference results.

Loads inference JSONs once, then loops over parameter combinations
for tracking + evaluation. Much faster than running dvc exp per combo.

Usage:
    uv run python scripts/sweep.py \
        --infer-dir data/02_intermediate/train \
        --data-dir data/01_raw/train \
        --output-dir data/08_reporting/sweep
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
from pathlib import Path

from src.data import is_wf_sequence
from src.detector import load_inference_results
from src.evaluator import compute_metrics
from src.tracker import SimpleTracker
from src.types import FrameResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLDS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
MIN_CONSECUTIVES = [1, 2, 3, 4, 5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep tracking parameters.")
    parser.add_argument(
        "--infer-dir",
        type=Path,
        required=True,
        help="Path to inference results directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to sequence data directory (for GT labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for sweep results CSV.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all inference results once
    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Loading %d inference files...", len(infer_files))

    all_data: list[tuple[bool, list]] = []
    for infer_path in infer_files:
        seq_id = infer_path.stem
        frames = load_inference_results(infer_path)
        gt = is_wf_sequence(args.data_dir / seq_id)
        all_data.append((gt, frames))
    logger.info("Loaded %d sequences.", len(all_data))

    # Sweep
    combos = list(
        itertools.product(CONFIDENCE_THRESHOLDS, IOU_THRESHOLDS, MIN_CONSECUTIVES)
    )
    logger.info("Running %d parameter combinations...", len(combos))

    rows = []
    for conf_thresh, iou_thresh, min_consec in combos:
        tracker = SimpleTracker(
            iou_threshold=iou_thresh,
            min_consecutive=min_consec,
        )

        results = []
        for gt, frames in all_data:
            # Filter by confidence
            filtered_frames = [
                FrameResult(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    detections=[
                        d for d in frame.detections if d.confidence >= conf_thresh
                    ],
                )
                for frame in frames
            ]

            is_alarm, _tracks, confirmed_idx = tracker.process_sequence(filtered_frames)
            total_dets = sum(len(f.detections) for f in filtered_frames)
            first_ts = filtered_frames[0].timestamp if filtered_frames else None
            confirmed_ts = (
                filtered_frames[confirmed_idx].timestamp
                if confirmed_idx is not None
                else None
            )

            results.append(
                {
                    "is_positive_gt": gt,
                    "is_positive_pred": is_alarm,
                    "num_detections_total": total_dets,
                    "confirmed_timestamp": (
                        confirmed_ts.isoformat() if confirmed_ts else None
                    ),
                    "first_timestamp": (first_ts.isoformat() if first_ts else None),
                }
            )

        metrics = compute_metrics(results)
        row = {
            "confidence_threshold": conf_thresh,
            "iou_threshold": iou_thresh,
            "min_consecutive": min_consec,
            **metrics,
        }
        rows.append(row)

    # Sort by F1 descending
    rows.sort(key=lambda r: -r["f1"])

    # Write CSV
    csv_path = args.output_dir / "sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d results to %s", len(rows), csv_path)

    # Print top 10
    logger.info("Top 10 by F1:")
    logger.info(
        "  %-6s %-6s %-6s | %-6s %-6s %-6s %-6s | %-8s",
        "conf",
        "iou",
        "minC",
        "P",
        "R",
        "F1",
        "FPR",
        "TTD(s)",
    )
    for row in rows[:10]:
        ttd = row.get("mean_ttd_seconds")
        ttd_str = f"{ttd:.0f}" if ttd is not None else "N/A"
        logger.info(
            "  %-6s %-6s %-6s | %-6s %-6s %-6s %-6s | %-8s",
            row["confidence_threshold"],
            row["iou_threshold"],
            row["min_consecutive"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["fpr"],
            ttd_str,
        )


if __name__ == "__main__":
    main()

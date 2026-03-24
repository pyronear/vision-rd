"""Grid-search tracking parameters on cached inference results.

Loads all inference JSONs once into memory, then sweeps over a grid of
parameter combinations using multiprocessing. For each combo the
SimpleTracker is run and sequence-level metrics are computed. Results are
written to a CSV sorted by F1 descending, and the top-10 configurations
are logged.

Usage:
    uv run python scripts/sweep.py \
        --infer-dir data/02_intermediate/train \
        --data-dir data/01_raw/datasets/train \
        --output-dir data/08_reporting/sweep

    # Include rule parameters in the sweep grid:
    uv run python scripts/sweep.py \
        --infer-dir data/02_intermediate/train \
        --data-dir data/01_raw/datasets/train \
        --output-dir data/08_reporting/sweep \
        --sweep-rules
"""

import argparse
import csv
import itertools
import logging
import multiprocessing
import os
from pathlib import Path

from tqdm import tqdm

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
MAX_DETECTION_AREAS: list[float | None] = [None, 0.05, 0.10, 0.15]
MAX_MISSES_VALUES = [0, 1, 2]
MIN_MEAN_CONFIDENCES = [0.0, 0.3, 0.4, 0.5]
MIN_AREA_CHANGES = [0.0, 1.0, 1.1, 1.2]

# Module-level variable shared across worker processes (set via _init_worker)
_all_data: list[tuple[bool, list[FrameResult]]] = []


def _init_worker(data: list[tuple[bool, list[FrameResult]]]) -> None:
    global _all_data
    _all_data = data


def _evaluate_combo(
    combo: tuple[float, float, int, float | None, int, float, float],
) -> dict:
    (
        conf_thresh,
        iou_thresh,
        min_consec,
        max_det_area,
        max_misses,
        min_mean_conf,
        min_area_change,
    ) = combo

    tracker = SimpleTracker(
        iou_threshold=iou_thresh,
        min_consecutive=min_consec,
        max_misses=max_misses,
        use_confidence_filter=min_mean_conf > 0,
        min_mean_confidence=min_mean_conf,
        use_area_change_filter=min_area_change > 0,
        min_area_change=min_area_change,
    )

    results = []
    for gt, frames in _all_data:
        filtered_frames = [
            FrameResult(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=[
                    d
                    for d in frame.detections
                    if d.confidence >= conf_thresh
                    and (max_det_area is None or d.w * d.h <= max_det_area)
                ],
            )
            for frame in frames
        ]

        is_alarm, _tracks, confirmed_idx = tracker.process_sequence(filtered_frames)
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
                "num_detections_total": sum(len(f.detections) for f in filtered_frames),
                "confirmed_timestamp": (
                    confirmed_ts.isoformat() if confirmed_ts else None
                ),
                "first_timestamp": first_ts.isoformat() if first_ts else None,
            }
        )

    metrics = compute_metrics(results)
    return {
        "confidence_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "min_consecutive": min_consec,
        "max_detection_area": max_det_area,
        "max_misses": max_misses,
        "min_mean_confidence": min_mean_conf,
        "min_area_change": min_area_change,
        **metrics,
    }


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
    parser.add_argument(
        "--sweep-rules",
        action="store_true",
        help="Also sweep over rule parameters (max_misses, confidence, area_change).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: number of CPUs).",
    )
    parser.add_argument(
        "--filter-prefix",
        type=str,
        default=None,
        help="Only include sequences whose ID starts with this prefix.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all inference results once
    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Loading %d inference files...", len(infer_files))

    all_data: list[tuple[bool, list[FrameResult]]] = []
    for infer_path in infer_files:
        seq_id = infer_path.stem
        if args.filter_prefix and not seq_id.startswith(args.filter_prefix):
            continue
        frames = load_inference_results(infer_path)
        gt = is_wf_sequence(args.data_dir / seq_id)
        all_data.append((gt, frames))
    logger.info("Loaded %d sequences.", len(all_data))

    # Build parameter grid
    if args.sweep_rules:
        combos = list(
            itertools.product(
                CONFIDENCE_THRESHOLDS,
                IOU_THRESHOLDS,
                MIN_CONSECUTIVES,
                MAX_DETECTION_AREAS,
                MAX_MISSES_VALUES,
                MIN_MEAN_CONFIDENCES,
                MIN_AREA_CHANGES,
            )
        )
    else:
        combos = list(
            itertools.product(
                CONFIDENCE_THRESHOLDS,
                IOU_THRESHOLDS,
                MIN_CONSECUTIVES,
                [None],
                [0],
                [0.0],
                [0.0],
            )
        )

    n_workers = args.workers or os.cpu_count() or 1
    logger.info(
        "Running %d parameter combinations with %d workers...",
        len(combos),
        n_workers,
    )

    with multiprocessing.Pool(
        processes=n_workers, initializer=_init_worker, initargs=(all_data,)
    ) as pool:
        rows = list(
            tqdm(
                pool.imap_unordered(_evaluate_combo, combos),
                total=len(combos),
                desc="Sweep",
            )
        )

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
    if args.sweep_rules:
        logger.info(
            "  %-6s %-6s %-4s %-8s %-4s %-8s %-8s | %-6s %-6s %-6s %-6s | %-6s",
            "conf",
            "iou",
            "mnC",
            "maxArea",
            "mxM",
            "mnConf",
            "mnArea",
            "P",
            "R",
            "F1",
            "FPR",
            "TTD",
        )
        for row in rows[:10]:
            ttd = row.get("mean_ttd_seconds")
            ttd_str = f"{ttd:.0f}" if ttd is not None else "N/A"
            max_area = row["max_detection_area"]
            max_area_str = f"{max_area}" if max_area is not None else "None"
            logger.info(
                "  %-6s %-6s %-4s %-8s %-4s %-8s %-8s | %-6s %-6s %-6s %-6s | %-6s",
                row["confidence_threshold"],
                row["iou_threshold"],
                row["min_consecutive"],
                max_area_str,
                row["max_misses"],
                row["min_mean_confidence"],
                row["min_area_change"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["fpr"],
                ttd_str,
            )
    else:
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

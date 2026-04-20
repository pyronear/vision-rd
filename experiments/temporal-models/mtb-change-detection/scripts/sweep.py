"""Grid-search MTB change detection + tracking parameters.

Pre-computes per-detection change ratios at multiple pixel thresholds
(loading images once), then sweeps over all parameter combinations using
multiprocessing. For each combo the detections are filtered by change
ratio, run through the SimpleTracker, and sequence-level metrics are
computed. Results are written to a CSV sorted by F1 descending.

Usage:
    uv run python scripts/sweep.py \
        --infer-dir data/03_primary/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/08_reporting/sweep/val/all

    # Include tracker rule parameters in the sweep:
    uv run python scripts/sweep.py \
        --infer-dir data/03_primary/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/08_reporting/sweep/val/all \
        --sweep-rules
"""

import argparse
import csv
import itertools
import logging
import multiprocessing
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mtb_change_detection.change_detector import compute_change_ratio_in_bbox
from mtb_change_detection.data import get_sorted_frames, is_wf_sequence
from mtb_change_detection.detector import load_inference_results
from mtb_change_detection.evaluator import compute_metrics
from mtb_change_detection.tracker import SimpleTracker
from mtb_change_detection.types import FrameResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Sweep grids ---
# Change detection (the main knobs)
PIXEL_THRESHOLDS = [3, 5, 7, 10, 15, 19, 25, 30, 40]
MIN_CHANGE_RATIOS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

# Tracker (minimal by default, expanded with --sweep-rules)
CONFIDENCE_THRESHOLDS = [0.01, 0.1, 0.2, 0.3]
IOU_THRESHOLDS = [0.1, 0.2, 0.3]
MIN_CONSECUTIVES = [1, 2, 3, 4, 5]
MAX_DETECTION_AREAS: list[float | None] = [None, 0.05, 0.10]
MAX_MISSES_VALUES = [0, 1]

# Per-detection change ratios, keyed by pixel_threshold.
# Structure: list of (gt, frames, change_data) per sequence.
# change_data[frame_idx] = list of {pixel_threshold: ratio} per detection.
_all_data: list[tuple[bool, list[FrameResult], list[list[dict[int, float]]]]] = []


def _init_worker(
    data: list[tuple[bool, list[FrameResult], list[list[dict[int, float]]]]],
) -> None:
    global _all_data
    _all_data = data


def _evaluate_combo(combo: tuple) -> dict:
    (
        pixel_threshold,
        min_change_ratio,
        conf_thresh,
        iou_thresh,
        min_consec,
        max_det_area,
        max_misses,
    ) = combo

    tracker = SimpleTracker(
        iou_threshold=iou_thresh,
        min_consecutive=min_consec,
        max_misses=max_misses,
    )

    results = []
    for gt, frames, change_data in _all_data:
        # Filter detections by confidence, area, and change ratio
        filtered_frames = []
        for frame_idx, frame in enumerate(frames):
            det_ratios = change_data[frame_idx]
            kept = []
            for det_idx, det in enumerate(frame.detections):
                if det.confidence < conf_thresh:
                    continue
                if max_det_area is not None and det.w * det.h > max_det_area:
                    continue
                ratio = det_ratios[det_idx].get(pixel_threshold, 0.0)
                if ratio >= min_change_ratio:
                    kept.append(det)
            filtered_frames.append(
                FrameResult(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    detections=kept,
                )
            )

        is_alarm, _tracks, confirmed_idx, _ft = tracker.process_sequence(
            filtered_frames
        )
        total_dets = sum(len(f.detections) for f in filtered_frames)

        results.append(
            {
                "is_positive_gt": gt,
                "is_positive_pred": is_alarm,
                "num_detections_total": total_dets,
                "confirmed_frame_index": confirmed_idx,
            }
        )

    metrics = compute_metrics(results)
    return {
        "pixel_threshold": pixel_threshold,
        "min_change_ratio": min_change_ratio,
        "confidence_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "min_consecutive": min_consec,
        "max_detection_area": max_det_area,
        "max_misses": max_misses,
        **metrics,
    }


def _find_sequence_dir(data_dir: Path, seq_id: str) -> Path | None:
    for category in ("wildfire", "fp"):
        candidate = data_dir / category / seq_id
        if candidate.is_dir():
            return candidate
    return None


def _precompute_change_ratios(
    frames: list[FrameResult],
    image_paths: list[Path],
    pixel_thresholds: list[int],
) -> list[list[dict[int, float]]]:
    """Pre-compute per-detection change ratios at all pixel thresholds.

    Returns a list (one per frame) of lists (one per detection) of dicts
    mapping pixel_threshold -> change_ratio.
    """
    image_by_id: dict[str, Path] = {p.stem: p for p in image_paths}
    result: list[list[dict[int, float]]] = []
    prev_gray: np.ndarray | None = None

    for fr in frames:
        img_path = image_by_id.get(fr.frame_id)
        curr_gray = (
            cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_path is not None
            else None
        )

        if prev_gray is None or curr_gray is None or prev_gray.shape != curr_gray.shape:
            # No change data for this frame
            result.append([{t: 0.0 for t in pixel_thresholds} for _ in fr.detections])
            prev_gray = curr_gray
            continue

        # Compute raw diff once (not thresholded)
        raw_diff = np.abs(curr_gray.astype(np.int16) - prev_gray.astype(np.int16))

        det_ratios: list[dict[int, float]] = []
        for det in fr.detections:
            ratios: dict[int, float] = {}
            for thresh in pixel_thresholds:
                change_mask = raw_diff > thresh
                ratios[thresh] = compute_change_ratio_in_bbox(
                    change_mask, det.cx, det.cy, det.w, det.h
                )
            det_ratios.append(ratios)

        result.append(det_ratios)
        prev_gray = curr_gray

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep MTB change detection + tracking parameters."
    )
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sweep-rules",
        action="store_true",
        help="Also sweep tracker rules (max_detection_area, max_misses).",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--filter-prefix", type=str, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all inference results and pre-compute change ratios
    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info(
        "Loading %d files and pre-computing change ratios...",
        len(infer_files),
    )

    all_data: list[tuple[bool, list[FrameResult], list[list[dict[int, float]]]]] = []
    for infer_path in tqdm(infer_files, desc="Pre-computing"):
        seq_id = infer_path.stem
        if args.filter_prefix and not seq_id.startswith(args.filter_prefix):
            continue
        frames = load_inference_results(infer_path)
        seq_dir = _find_sequence_dir(args.data_dir, seq_id)
        gt = is_wf_sequence(seq_dir) if seq_dir is not None else False
        image_paths = get_sorted_frames(seq_dir) if seq_dir is not None else []

        change_data = _precompute_change_ratios(frames, image_paths, PIXEL_THRESHOLDS)
        all_data.append((gt, frames, change_data))
    logger.info("Loaded %d sequences.", len(all_data))

    # Build parameter grid
    if args.sweep_rules:
        combos = list(
            itertools.product(
                PIXEL_THRESHOLDS,
                MIN_CHANGE_RATIOS,
                CONFIDENCE_THRESHOLDS,
                IOU_THRESHOLDS,
                MIN_CONSECUTIVES,
                MAX_DETECTION_AREAS,
                MAX_MISSES_VALUES,
            )
        )
    else:
        combos = list(
            itertools.product(
                PIXEL_THRESHOLDS,
                MIN_CHANGE_RATIOS,
                [0.01],  # pass-through
                [0.1],
                MIN_CONSECUTIVES,
                [None],
                [0],
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
    logger.info(
        "  %-6s %-8s %-6s %-6s %-4s %-8s %-4s | %-6s %-6s %-6s %-6s | %-6s",
        "pxThr",
        "minChR",
        "conf",
        "iou",
        "mnC",
        "maxArea",
        "mxM",
        "P",
        "R",
        "F1",
        "FPR",
        "TTD",
    )
    for row in rows[:10]:
        ttd = row.get("mean_ttd_frames")
        ttd_str = f"{ttd:.1f}" if ttd is not None else "N/A"
        max_area = row["max_detection_area"]
        max_area_str = f"{max_area}" if max_area is not None else "None"
        logger.info(
            "  %-6s %-8s %-6s %-6s %-4s %-8s %-4s | %-6s %-6s %-6s %-6s | %-6s",
            row["pixel_threshold"],
            row["min_change_ratio"],
            row["confidence_threshold"],
            row["iou_threshold"],
            row["min_consecutive"],
            max_area_str,
            row["max_misses"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["fpr"],
            ttd_str,
        )


if __name__ == "__main__":
    main()

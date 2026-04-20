"""Grid-search predictor temporal parameters on cached inference results.

Loads per-frame YOLO detections from the infer stage, then sweeps over a
grid of (conf_thresh, nb_consecutive_frames) combinations using
multiprocessing.  For each combo the Predictor's temporal logic is replayed
and sequence-level metrics are computed.  Results are written to a CSV
sorted by F1 descending, and the top-10 configurations are logged.

Usage:
    uv run python scripts/sweep.py \
        --infer-dir data/03_primary/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/08_reporting/sweep/val/all

    # Pyronear-only subset:
    uv run python scripts/sweep.py \
        --infer-dir data/03_primary/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/08_reporting/sweep/val/pyronear \
        --filter-prefix pyronear
"""

import argparse
import csv
import itertools
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pyro_detector_baseline.data import is_wf_sequence
from pyro_detector_baseline.evaluator import compute_metrics
from pyro_detector_baseline.predictor_wrapper import (
    create_replay_predictor,
    load_detections,
    replay_sequence,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONF_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
DEFAULT_NB_CONSECUTIVE_FRAMES = [2, 3, 4, 5, 6, 7, 8, 10]

# Module-level shared state set via _init_worker
_all_sequences: list[tuple[str, bool, list[tuple[str, np.ndarray]], str]] = []


def _init_worker(
    sequences: list[tuple[str, bool, list[tuple[str, np.ndarray]], str]],
) -> None:
    global _all_sequences
    _all_sequences = sequences


def _evaluate_combo(combo: tuple[float, int]) -> dict:
    conf_thresh, nb_frames = combo
    replay = create_replay_predictor(conf_thresh, nb_frames)

    results: list[dict] = []
    for seq_id, is_positive_gt, frame_detections in _all_sequences:
        trigger_idx, _confidences = replay_sequence(replay, frame_detections, seq_id)

        results.append(
            {
                "is_positive_gt": is_positive_gt,
                "is_positive_pred": trigger_idx is not None,
                "confirmed_frame_index": trigger_idx,
            }
        )

    metrics = compute_metrics(results)
    return {
        "conf_thresh": conf_thresh,
        "nb_consecutive_frames": nb_frames,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep predictor temporal parameters.")
    parser.add_argument(
        "--infer-dir",
        type=Path,
        required=True,
        help="Path to cached inference results directory.",
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
    parser.add_argument(
        "--conf-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_CONF_THRESHOLDS,
        help="Confidence thresholds to sweep (default: %(default)s).",
    )
    parser.add_argument(
        "--nb-frames-values",
        type=int,
        nargs="+",
        default=DEFAULT_NB_CONSECUTIVE_FRAMES,
        help="Sliding window sizes to sweep (default: %(default)s).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all cached inference results
    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Loading %d inference files...", len(infer_files))

    all_sequences: list[tuple[str, bool, list[tuple[str, np.ndarray]], str]] = []
    for infer_path in infer_files:
        seq_id = infer_path.stem
        if args.filter_prefix and not seq_id.startswith(args.filter_prefix):
            continue

        seq_dir = args.data_dir / seq_id
        # Handle nested layout (wildfire/seq_id or fp/seq_id)
        if not seq_dir.is_dir():
            seq_dir_candidates = list(args.data_dir.glob(f"*/{seq_id}"))
            if not seq_dir_candidates:
                logger.warning("No data dir for %s, skipping.", seq_id)
                continue
            seq_dir = seq_dir_candidates[0]

        frame_detections = load_detections(infer_path)
        if not frame_detections:
            continue

        gt = is_wf_sequence(seq_dir)
        all_sequences.append((seq_id, gt, frame_detections))

    logger.info("Loaded %d sequences.", len(all_sequences))

    if not all_sequences:
        logger.error("No sequences found — nothing to sweep.")
        return

    # Build parameter grid
    combos = list(itertools.product(args.conf_thresholds, args.nb_frames_values))
    n_workers = args.workers or os.cpu_count() or 1
    logger.info(
        "Running %d parameter combinations with %d workers...",
        len(combos),
        n_workers,
    )

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(all_sequences,),
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
        "  %-6s %-6s | %-6s %-6s %-6s %-6s | %-8s",
        "conf",
        "nbF",
        "P",
        "R",
        "F1",
        "FPR",
        "TTD(fr)",
    )
    for row in rows[:10]:
        ttd = row.get("mean_ttd_frames")
        ttd_str = f"{ttd:.1f}" if ttd is not None else "N/A"
        logger.info(
            "  %-6s %-6s | %-6s %-6s %-6s %-6s | %-8s",
            row["conf_thresh"],
            row["nb_consecutive_frames"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["fpr"],
            ttd_str,
        )


if __name__ == "__main__":
    main()

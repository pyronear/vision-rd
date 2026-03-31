"""Ablation study: measure individual and combined impact of tracking rules.

Loads cached inference results once, then runs 8 combinations of the three
toggleable rules (gap tolerance, confidence filter, area-change filter) on
top of fixed base parameters. Outputs a comparison table as CSV and to stdout.

Usage:
    uv run python scripts/ablation.py \
        --infer-dir data/02_intermediate/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/08_reporting/ablation_val \
        --confidence-threshold 0.4 \
        --iou-threshold 0.1 \
        --min-consecutive 2
"""

import argparse
import csv
import itertools
import logging
from pathlib import Path

from tracking_fsm_baseline.data import is_wf_sequence
from tracking_fsm_baseline.detector import load_inference_results
from tracking_fsm_baseline.evaluator import evaluate_tracker
from tracking_fsm_baseline.tracker import SimpleTracker
from tracking_fsm_baseline.types import FrameResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RULES = ["gap", "confidence", "area_change"]


def _run_combo(
    all_data: list[tuple[bool, list[FrameResult]]],
    conf_thresh: float,
    iou_thresh: float,
    min_consecutive: int,
    max_detection_area: float | None,
    use_gap: bool,
    max_misses: int,
    use_confidence: bool,
    min_mean_confidence: float,
    use_area_change: bool,
    min_area_change: float,
) -> dict:
    tracker = SimpleTracker(
        iou_threshold=iou_thresh,
        min_consecutive=min_consecutive,
        max_misses=max_misses if use_gap else 0,
        use_confidence_filter=use_confidence,
        min_mean_confidence=min_mean_confidence,
        use_area_change_filter=use_area_change,
        min_area_change=min_area_change,
    )

    _results, metrics = evaluate_tracker(
        tracker, all_data, conf_thresh, max_detection_area
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for tracking rules.")
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--min-consecutive", type=int, default=2)
    parser.add_argument("--max-detection-area", type=float, default=None)
    parser.add_argument("--max-misses", type=int, default=2)
    parser.add_argument("--min-mean-confidence", type=float, default=0.4)
    parser.add_argument("--min-area-change", type=float, default=1.1)
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

    # Run all 8 combinations of (gap, confidence, area_change) on/off
    rows = []
    for use_gap, use_conf, use_area in itertools.product([False, True], repeat=3):
        label_parts = []
        if use_gap:
            label_parts.append("gap")
        if use_conf:
            label_parts.append("confidence")
        if use_area:
            label_parts.append("area_change")
        label = "+".join(label_parts) if label_parts else "baseline"

        metrics = _run_combo(
            all_data=all_data,
            conf_thresh=args.confidence_threshold,
            iou_thresh=args.iou_threshold,
            min_consecutive=args.min_consecutive,
            max_detection_area=args.max_detection_area,
            use_gap=use_gap,
            max_misses=args.max_misses,
            use_confidence=use_conf,
            min_mean_confidence=args.min_mean_confidence,
            use_area_change=use_area,
            min_area_change=args.min_area_change,
        )

        row = {"rules": label, **metrics}
        rows.append(row)

    # Write CSV
    csv_path = args.output_dir / "ablation_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved ablation results to %s", csv_path)

    # Print table
    header = f"{'Rules':<30} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'TTD':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for row in rows:
        ttd = row.get("mean_ttd_seconds")
        ttd_str = f"{ttd:.0f}s" if ttd is not None else "N/A"
        logger.info(
            "%-30s %6.3f %6.3f %6.3f %6.3f %8s",
            row["rules"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["fpr"],
            ttd_str,
        )


if __name__ == "__main__":
    main()

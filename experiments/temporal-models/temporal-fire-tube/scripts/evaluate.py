"""Evaluate fire-tube prediction results against ground truth.

Loads prediction_results.json, computes sequence-level metrics, and
generates comparison plots against a YOLO-only baseline.

Usage:
    uv run python scripts/evaluate.py \
        --results-dir data/07_model_output/val \
        --output-dir data/08_reporting/val/all
"""

import argparse
import json
import logging
from pathlib import Path

from src.evaluator import (
    compute_metrics,
    compute_yolo_only_baseline,
    load_prediction_results,
    plot_comparison,
    plot_confusion_matrix,
    plot_confusion_matrix_percentages,
    plot_ttd_histogram,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fire-tube results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to prediction results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for metrics and plots.",
    )
    parser.add_argument(
        "--filter-prefix",
        type=str,
        default=None,
        help="Only include sequences whose ID starts with this prefix.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = load_prediction_results(args.results_dir / "prediction_results.json")
    if args.filter_prefix:
        results = [
            r for r in results if r["sequence_id"].startswith(args.filter_prefix)
        ]
    logger.info("Loaded %d sequence results.", len(results))

    # Compute metrics
    fire_tube_metrics = compute_metrics(results)
    yolo_metrics = compute_yolo_only_baseline(results)

    # Save metrics
    all_metrics = {
        "yolo_only": yolo_metrics,
        "fire_tube": fire_tube_metrics,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    logger.info("Saved metrics to %s", metrics_path)

    # Print summary
    logger.info("=== YOLO-only Baseline ===")
    logger.info(
        "  P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f",
        yolo_metrics["precision"],
        yolo_metrics["recall"],
        yolo_metrics["f1"],
        yolo_metrics["fpr"],
    )
    logger.info("=== Fire-Tube ===")
    logger.info(
        "  P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f",
        fire_tube_metrics["precision"],
        fire_tube_metrics["recall"],
        fire_tube_metrics["f1"],
        fire_tube_metrics["fpr"],
    )
    if fire_tube_metrics["mean_ttd_seconds"] is not None:
        logger.info(
            "  Mean TTD=%.1fs  Median TTD=%.1fs",
            fire_tube_metrics["mean_ttd_seconds"],
            fire_tube_metrics["median_ttd_seconds"],
        )

    # Generate plots
    plot_confusion_matrix(fire_tube_metrics, plots_dir / "confusion_matrix.png")
    plot_confusion_matrix_percentages(
        fire_tube_metrics, plots_dir / "confusion_matrix_percentages.png"
    )
    plot_comparison(yolo_metrics, fire_tube_metrics, plots_dir / "comparison.png")
    plot_ttd_histogram(results, plots_dir / "ttd_histogram.png")
    logger.info("Saved plots to %s", plots_dir)


if __name__ == "__main__":
    main()

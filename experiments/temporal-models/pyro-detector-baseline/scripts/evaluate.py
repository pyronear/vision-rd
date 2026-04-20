"""Evaluate prediction results against ground truth.

Loads tracking_results.json, computes sequence-level metrics (precision,
recall, F1, FPR, time-to-detection) for both the Predictor and a
single-frame baseline (any detection = alarm).  Generates plots:
confusion matrix, metric comparison bar chart, and TTD histogram.

Usage:
    uv run python scripts/evaluate.py \
        --track-dir data/07_model_output/val \
        --output-dir data/08_reporting/val/all
"""

import argparse
import json
import logging
from pathlib import Path

from pyro_detector_baseline.evaluator import (
    compute_metrics,
    compute_single_frame_baseline,
    load_tracking_results,
    plot_comparison,
    plot_confusion_matrix,
    plot_confusion_matrix_percentages,
    plot_ttd_histogram,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction results.")
    parser.add_argument(
        "--track-dir",
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

    results = load_tracking_results(args.track_dir / "tracking_results.json")
    if args.filter_prefix:
        results = [
            r for r in results if r["sequence_id"].startswith(args.filter_prefix)
        ]
    logger.info("Loaded %d sequence results.", len(results))

    # Compute metrics
    predictor_metrics = compute_metrics(results)
    baseline_metrics = compute_single_frame_baseline(results)

    # Save metrics
    all_metrics = {
        "single_frame": baseline_metrics,
        "predictor": predictor_metrics,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    logger.info("Saved metrics to %s", metrics_path)

    # Print summary
    logger.info("=== Single-frame Baseline ===")
    logger.info(
        "  P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f",
        baseline_metrics["precision"],
        baseline_metrics["recall"],
        baseline_metrics["f1"],
        baseline_metrics["fpr"],
    )
    logger.info("=== Predictor (temporal) ===")
    logger.info(
        "  P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f",
        predictor_metrics["precision"],
        predictor_metrics["recall"],
        predictor_metrics["f1"],
        predictor_metrics["fpr"],
    )
    if predictor_metrics["mean_ttd_frames"] is not None:
        logger.info(
            "  Mean TTD=%.1f frames  Median TTD=%.1f frames",
            predictor_metrics["mean_ttd_frames"],
            predictor_metrics["median_ttd_frames"],
        )

    # Generate plots
    plot_confusion_matrix(predictor_metrics, plots_dir / "confusion_matrix.png")
    plot_confusion_matrix_percentages(
        predictor_metrics, plots_dir / "confusion_matrix_percentages.png"
    )
    plot_comparison(baseline_metrics, predictor_metrics, plots_dir / "comparison.png")
    plot_ttd_histogram(results, plots_dir / "ttd_histogram.png")
    logger.info("Saved plots to %s", plots_dir)


if __name__ == "__main__":
    main()

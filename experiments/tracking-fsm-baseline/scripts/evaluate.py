"""Evaluate tracking results and generate metrics + plots."""

import argparse
import json
import logging
from pathlib import Path

from src.evaluator import (
    compute_metrics,
    compute_yolo_only_baseline,
    load_tracking_results,
    plot_comparison,
    plot_confusion_matrix,
    plot_ttd_histogram,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tracking results.")
    parser.add_argument(
        "--track-dir",
        type=Path,
        required=True,
        help="Path to tracking results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for metrics and plots.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = load_tracking_results(args.track_dir / "tracking_results.json")
    logger.info("Loaded %d sequence results.", len(results))

    # Compute metrics
    tracking_metrics = compute_metrics(results)
    yolo_metrics = compute_yolo_only_baseline(results)

    # Save metrics
    all_metrics = {
        "yolo_only": yolo_metrics,
        "tracking": tracking_metrics,
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
    logger.info("=== Tracking (min_consecutive) ===")
    logger.info(
        "  P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f",
        tracking_metrics["precision"],
        tracking_metrics["recall"],
        tracking_metrics["f1"],
        tracking_metrics["fpr"],
    )
    if tracking_metrics["mean_ttd_seconds"] is not None:
        logger.info(
            "  Mean TTD=%.1fs  Median TTD=%.1fs",
            tracking_metrics["mean_ttd_seconds"],
            tracking_metrics["median_ttd_seconds"],
        )

    # Generate plots
    plot_confusion_matrix(tracking_metrics, plots_dir / "confusion_matrix.png")
    plot_comparison(yolo_metrics, tracking_metrics, plots_dir / "comparison.png")
    plot_ttd_histogram(results, plots_dir / "ttd_histogram.png")
    logger.info("Saved plots to %s", plots_dir)


if __name__ == "__main__":
    main()

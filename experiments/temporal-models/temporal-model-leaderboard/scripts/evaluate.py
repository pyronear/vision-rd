"""Evaluate a temporal model on the pyro-dataset test set.

Loads a model package, runs it on every test sequence, and writes
per-sequence results and aggregated metrics to the output directory.

Usage:
    uv run python scripts/evaluate.py \
        --model-name fsm-tracking-baseline \
        --model-type fsm-tracking-baseline \
        --model-package data/01_raw/models/fsm-tracking-baseline.zip \
        --test-dir data/01_raw/sequential_test/test \
        --output-dir data/07_model_output/fsm-tracking-baseline
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from temporal_model_leaderboard.metrics import compute_metrics
from temporal_model_leaderboard.registry import MODEL_REGISTRY, load_model
from temporal_model_leaderboard.runner import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a temporal model on the test set."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Human-readable model name for the leaderboard.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help="Model type key (selects the model class to load).",
    )
    parser.add_argument(
        "--model-package",
        type=Path,
        required=True,
        help="Path to the model package (.zip).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Path to the test split root (e.g., sequential_test/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results and metrics.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s from %s", args.model_type, args.model_package)
    model = load_model(args.model_type, args.model_package)

    logger.info("Evaluating on %s", args.test_dir)
    results = evaluate_model(model, args.test_dir)
    metrics = compute_metrics(args.model_name, results)

    results_path = args.output_dir / "results.json"
    results_data = [dataclasses.asdict(r) for r in results]
    results_path.write_text(json.dumps(results_data, indent=2))
    logger.info("Saved %d results to %s", len(results), results_path)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(dataclasses.asdict(metrics), indent=2))
    logger.info("Saved metrics to %s", metrics_path)

    logger.info(
        "  P=%.4f  R=%.4f  F1=%.4f  FPR=%.4f",
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.fpr,
    )
    if metrics.mean_ttd_frames is not None:
        logger.info(
            "  Mean TTD=%.1f frames  Median TTD=%.1f frames",
            metrics.mean_ttd_frames,
            metrics.median_ttd_frames,
        )


if __name__ == "__main__":
    main()

"""Aggregate per-model metrics into a ranked leaderboard.

Reads metrics.json from each model's output directory and produces
a sorted leaderboard table and JSON.

Usage:
    uv run python scripts/leaderboard.py \
        --results-dir data/07_model_output \
        --output-dir data/08_reporting
"""

import argparse
import json
import logging
from pathlib import Path

from temporal_model_leaderboard.leaderboard import format_table, sort_entries, to_json
from temporal_model_leaderboard.types import LeaderboardEntry, ModelMetrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_metrics(metrics_path: Path) -> ModelMetrics:
    data = json.loads(metrics_path.read_text())
    return ModelMetrics(**data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce a leaderboard from model evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing per-model subdirectories with metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for leaderboard files.",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="f1",
        help="Metric to sort by (default: f1).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[LeaderboardEntry] = []
    for model_dir in sorted(args.results_dir.iterdir()):
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _load_metrics(metrics_path)
        entries.append(LeaderboardEntry(metrics=metrics))
        logger.info("Loaded metrics for %s", metrics.model_name)

    if not entries:
        logger.warning("No model results found in %s", args.results_dir)
        return

    sorted_entries = sort_entries(entries, args.primary_metric)

    table = format_table(sorted_entries)
    table_path = args.output_dir / "leaderboard.txt"
    table_path.write_text(table + "\n")
    logger.info("Saved leaderboard table to %s", table_path)

    json_str = to_json(sorted_entries)
    json_path = args.output_dir / "leaderboard.json"
    json_path.write_text(json_str + "\n")
    logger.info("Saved leaderboard JSON to %s", json_path)

    print()
    print(table)


if __name__ == "__main__":
    main()

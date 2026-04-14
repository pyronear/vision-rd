"""Render per-run training curves from a Lightning CSVLogger directory."""

import argparse
import re
import sys
from pathlib import Path

from smokeynet_adapted.training_plots import plot_training_curves


def _find_latest_metrics_csv(csv_log_dir: Path) -> Path:
    version_dirs = []
    for child in csv_log_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"version_(\d+)", child.name)
        if match:
            version_dirs.append((int(match.group(1)), child))
    if not version_dirs:
        raise FileNotFoundError(f"No version_* directories under {csv_log_dir}")
    version_dirs.sort(key=lambda pair: pair[0])
    latest_dir = version_dirs[-1][1]
    csv_path = latest_dir / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No metrics.csv at {csv_path}")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render training curves as a 2x3 PNG from a Lightning "
            "CSVLogger directory. When multiple version_* subdirs "
            "exist, the highest-numbered one is used."
        )
    )
    parser.add_argument("--csv-log-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    args = parser.parse_args()

    csv_path = _find_latest_metrics_csv(args.csv_log_dir)
    plot_training_curves(csv_path, args.output_path, args.title)
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

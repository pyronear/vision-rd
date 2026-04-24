"""Validate reviewer-applied tags in FiftyOne without touching disk.

Mid-session sanity check: walks every ``dq-frame_*`` dataset (or a
specified one), prints per-view review progress, and runs the same
validator that :mod:`scripts.export_review` would enforce at export
time. Exits 0 on clean input, 1 on any invalid tag.

Usage::

    uv run --group explore python scripts/validate_review.py

    # Only a specific dataset:
    uv run --group explore python scripts/validate_review.py \\
        --dataset dq-frame_yolo11s-nimble-narwhal_val
"""

import argparse
import logging
import sys

import fiftyone as fo
from export_review import collect_stem_tags, iter_stem_tags

from data_quality_frame_level.fiftyone_build import FN_VIEW_NAME, FP_VIEW_NAME
from data_quality_frame_level.review import (
    count_reviewed,
    format_invalid_report,
    format_progress_line,
    scan_invalid,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Only validate this single dataset (default: every dq-frame_*).",
    )
    return parser.parse_args()


def _print_progress(dataset_name: str) -> None:
    """Print per-view progress lines for one dataset."""
    dataset = fo.load_dataset(dataset_name)
    print(dataset_name)
    for kind, view_name in (("fp", FP_VIEW_NAME), ("fn", FN_VIEW_NAME)):
        view = dataset.load_saved_view(view_name)
        stem_tags = iter_stem_tags(view)
        reviewed, total = count_reviewed(stem_tags)
        print("  " + format_progress_line(kind, reviewed, total))


def main() -> None:
    args = _parse_args()
    if args.dataset is not None:
        names = [args.dataset]
    else:
        names = fo.list_datasets(glob_patt="dq-frame_*")
    if not names:
        raise SystemExit("No dq-frame_* datasets found. Run 'dvc repro' first.")

    # Progress first — useful even when there are no typos.
    for name in sorted(names):
        _print_progress(name)
    print()

    reports: list[str] = []
    for name in sorted(names):
        bad = scan_invalid(collect_stem_tags(name))
        if bad:
            reports.append(format_invalid_report(name, bad))

    if reports:
        print("\n\n".join(reports), file=sys.stderr)
        raise SystemExit(1)

    logger.info("Reviewer tags on %d dataset(s) are all valid.", len(names))


if __name__ == "__main__":
    main()

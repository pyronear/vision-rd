"""Dump FiftyOne sample tags to disk for persistence + DVC tracking.

Walks every ``dq-frame_*`` dataset, validates that every reviewer-applied
tag is in :data:`data_quality_frame_level.review.REVIEW_VOCAB` (or a
``reviewer:<handle>`` attribution), and writes one ``tags.json`` per
(model, split) under ``data/09_review/<model>/<split>/``. Stems whose
tag list is empty are omitted so the file only records actual review
decisions.

**Hard gate**: if any sample carries an unknown tag (typo, wrong case,
rogue prefix), no files are written — the script prints a report with
suggestions and exits non-zero. Fix the tags in the FiftyOne UI and re-run.

Usage::

    uv run --group explore python scripts/export_review.py

    # Only a specific dataset:
    uv run --group explore python scripts/export_review.py \\
        --dataset dq-frame_yolo11s-nimble-narwhal_val
"""

import argparse
import json
import logging
import re
from pathlib import Path

import fiftyone as fo

from data_quality_frame_level.review import (
    format_invalid_report,
    is_vocab_seed,
    payload_from_stem_tags,
    scan_invalid,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DQ_FRAME_PATTERN = re.compile(r"^dq-frame_(?P<model>.+)_(?P<split>train|val|test)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Only export this single dataset (default: every dq-frame_*).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/09_review"),
        help="Root directory for tag files (default: data/09_review).",
    )
    return parser.parse_args()


def _resolve_target_path(output_root: Path, dataset_name: str) -> Path:
    match = DQ_FRAME_PATTERN.match(dataset_name)
    if not match:
        raise SystemExit(
            f"Dataset '{dataset_name}' does not match dq-frame_<model>_<split>; "
            "can't infer model/split for the output path."
        )
    model = match.group("model")
    split = match.group("split")
    return output_root / model / split / "tags.json"


def collect_stem_tags(dataset_name: str) -> dict[str, list[str]]:
    """Return ``{stem: [tags]}`` for every non-seed sample in the dataset."""
    dataset = fo.load_dataset(dataset_name)
    stem_tags: dict[str, list[str]] = {}
    for sample in dataset:
        tags = list(sample.tags)
        if is_vocab_seed(tags):
            # Stale state from an earlier seeding experiment — skip it.
            continue
        stem = Path(sample.filepath).stem
        stem_tags[stem] = tags
    return stem_tags


def _write_one(
    dataset_name: str, stem_tags: dict[str, list[str]], output_root: Path
) -> int:
    payload = payload_from_stem_tags(dataset_name, stem_tags)
    target = _resolve_target_path(output_root, dataset_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tagged = len(payload["tags_by_stem"])
    logger.info("Exported %s → %s (%d tagged samples)", dataset_name, target, tagged)
    return tagged


def main() -> None:
    args = _parse_args()
    if args.dataset is not None:
        names = [args.dataset]
    else:
        names = fo.list_datasets(glob_patt="dq-frame_*")
    if not names:
        raise SystemExit("No dq-frame_* datasets found. Run 'dvc repro' first.")

    # Phase 1: scan all datasets and validate. No files written yet.
    scanned: dict[str, dict[str, list[str]]] = {}
    reports: list[str] = []
    for name in sorted(names):
        stem_tags = collect_stem_tags(name)
        scanned[name] = stem_tags
        bad = scan_invalid(stem_tags)
        if bad:
            reports.append(format_invalid_report(name, bad))

    if reports:
        raise SystemExit(
            "Review export refused: found invalid reviewer tags. "
            "Fix them in the FiftyOne app and re-run.\n\n"
            + "\n\n".join(reports)
        )

    # Phase 2: everything clean, write every file atomically.
    total = 0
    for name in sorted(scanned):
        total += _write_one(name, scanned[name], args.output_root)
    logger.info("Total tagged samples across %d dataset(s): %d", len(names), total)


if __name__ == "__main__":
    main()

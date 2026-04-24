"""Launch the FiftyOne app to browse the frame-level review datasets.

Opens the FiftyOne web UI on port 5151. Defaults to the FP review
queue on the ``_val`` dataset, sorted by predicted confidence
descending so the highest-confidence labeling candidates appear
first. Switch datasets in the FiftyOne sidebar; change the kind via
``--kind``.

Usage::

    # FPs on val, sorted by predicted confidence (default):
    uv run --group explore python scripts/launch_fiftyone.py

    # FNs on val, sorted by GT bbox area (largest first):
    uv run --group explore python scripts/launch_fiftyone.py --kind fn

    # All samples on val, unsorted:
    uv run --group explore python scripts/launch_fiftyone.py --kind all

    # Pin a specific dataset (bypasses the val auto-select):
    uv run --group explore python scripts/launch_fiftyone.py \\
        --dataset dq-frame_yolo11s-nimble-narwhal_test
"""

import argparse
import json
import logging
import re
import signal
import sys
from pathlib import Path

import fiftyone as fo

from data_quality_frame_level.fiftyone_build import FN_VIEW_NAME, FP_VIEW_NAME
from data_quality_frame_level.review import (
    REVIEW_VOCAB,
    REVIEWER_TAG_PREFIX,
    invalid_tags,
    merge_tags,
    stem_tags_from_payload,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DQ_FRAME_PATTERN = re.compile(r"^dq-frame_(?P<model>.+)_(?P<split>train|val|test)$")
DEFAULT_REVIEW_ROOT = Path("data/09_review")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=["fp", "fn", "all"],
        default="fp",
        help="Which review queue to surface first (default: fp).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Pin a specific dq-frame_* dataset (default: first *_val found).",
    )
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=DEFAULT_REVIEW_ROOT,
        help=(
            "Root directory of persisted review tags; tags.json files under "
            "<review-root>/<model>/<split>/ are auto-imported into the "
            "dataset before launch. Pass a missing path to skip."
        ),
    )
    return parser.parse_args()


def _pick_dataset(requested: str | None) -> str:
    datasets = fo.list_datasets(glob_patt="dq-frame_*")
    if not datasets:
        raise SystemExit(
            "No dq-frame_* datasets found. "
            "Run 'uv run dvc repro' (or the build_fiftyone stages) first."
        )
    if requested is not None:
        if requested not in datasets:
            available = ", ".join(sorted(datasets))
            raise SystemExit(f"Dataset '{requested}' not found. Available: {available}")
        return requested
    # Default: first _val dataset (usually the most useful starting point).
    return next((d for d in sorted(datasets) if d.endswith("_val")), datasets[0])


def _tag_file_for(review_root: Path, dataset_name: str) -> Path | None:
    match = DQ_FRAME_PATTERN.match(dataset_name)
    if not match:
        return None
    return review_root / match.group("model") / match.group("split") / "tags.json"


def _import_persisted_tags(dataset: fo.Dataset, tag_file: Path) -> int:
    """Merge tags from ``tag_file`` into ``dataset``; return samples touched.

    Defensively strips any tag outside the review vocabulary before
    merging — covers stale seed markers, hand-edited tags.json files,
    and tags from a future vocab extension. A WARNING is logged per
    affected stem so reviewers can see what was dropped.
    """
    payload = json.loads(tag_file.read_text())
    stem_to_tags = stem_tags_from_payload(payload)
    if not stem_to_tags:
        return 0
    touched = 0
    for sample in dataset:
        stem = Path(sample.filepath).stem
        incoming = stem_to_tags.get(stem)
        if not incoming:
            continue
        stripped = invalid_tags(incoming)
        if stripped:
            logger.warning(
                "Stripping unknown tag(s) on '%s' while importing %s: %s",
                stem,
                tag_file,
                stripped,
            )
            incoming = [t for t in incoming if t not in stripped]
        if not incoming:
            continue
        merged = merge_tags(list(sample.tags), incoming)
        if merged != list(sample.tags):
            sample.tags = merged
            sample.save()
            touched += 1
    return touched


def _print_vocab() -> None:
    """Log the review vocabulary on launch so reviewers can copy exact strings."""
    vocab_line = ", ".join(REVIEW_VOCAB)
    logger.info(
        "Review tag vocabulary (autocomplete trains after first exact match):"
    )
    logger.info("  %s", vocab_line)
    logger.info(
        "  plus %s<handle> for attribution (free-form).", REVIEWER_TAG_PREFIX
    )
    logger.info("Run 'make review-check' mid-session to catch typos early.")


def _build_view(dataset: fo.Dataset, kind: str):
    """Return the FP / FN review view, loaded from the dataset's saved views.

    The ``fp-by-confidence`` and ``fn-by-area`` saved views are persisted
    by :func:`data_quality_frame_level.fiftyone_build.build_dataset` at
    DVC-repro time. Using saved views (rather than rebuilding the view
    here) means the view is also available in the FiftyOne sidebar's
    "Saved views" dropdown when switching datasets.
    """
    if kind == "fp":
        return dataset.load_saved_view(FP_VIEW_NAME)
    if kind == "fn":
        return dataset.load_saved_view(FN_VIEW_NAME)
    return dataset.view()


def main() -> None:
    args = _parse_args()
    _print_vocab()
    dataset_name = _pick_dataset(args.dataset)
    dataset = fo.load_dataset(dataset_name)

    tag_file = _tag_file_for(args.review_root, dataset_name)
    if tag_file is not None and tag_file.is_file():
        touched = _import_persisted_tags(dataset, tag_file)
        logger.info(
            "Imported review tags from %s (%d samples updated)", tag_file, touched
        )

    view = _build_view(dataset, args.kind)

    print(
        f"Launching FiftyOne on :{args.port} with '{dataset_name}' "
        f"(kind={args.kind}, samples in view={len(view)})"
    )
    all_datasets = fo.list_datasets(glob_patt="dq-frame_*")
    print(f"Available datasets: {', '.join(sorted(all_datasets))}")

    # Force label + confidence to render as persistent overlays on every
    # bbox (in grid and expanded views). FiftyOne's defaults are True but
    # user-level configs can override them; set them on the session to be
    # certain.
    app_config = fo.app_config.copy()
    app_config.show_confidence = True
    app_config.show_label = True
    app_config.show_index = False
    app_config.show_tooltip = True

    session = fo.launch_app(view, port=args.port, config=app_config)

    def _shutdown(signum, _frame):
        print("\nShutting down FiftyOne...")
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    session.wait()


if __name__ == "__main__":
    main()

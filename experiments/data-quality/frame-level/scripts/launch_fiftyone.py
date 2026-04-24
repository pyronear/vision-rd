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
import signal
import sys

import fiftyone as fo

from data_quality_frame_level.fiftyone_build import FN_VIEW_NAME, FP_VIEW_NAME


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
            raise SystemExit(
                f"Dataset '{requested}' not found. Available: {available}"
            )
        return requested
    # Default: first _val dataset (usually the most useful starting point).
    return next((d for d in sorted(datasets) if d.endswith("_val")), datasets[0])


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
    dataset_name = _pick_dataset(args.dataset)
    dataset = fo.load_dataset(dataset_name)
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

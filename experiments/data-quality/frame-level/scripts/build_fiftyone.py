"""Build a FiftyOne review dataset for one (model, split) pair.

Loads GT from the split directory and predictions from predictions.json,
creates a persistent FiftyOne dataset with GT + predictions attached,
runs ``evaluate_detections``, and writes a sentinel JSON plus a summary.

Usage::

    uv run --group explore python scripts/build_fiftyone.py \\
        --dataset-name dq-frame_yolo11s-nimble-narwhal_train \\
        --split-dir data/01_raw/datasets/train \\
        --predictions <path>/predictions.json \\
        --iou-thresh 0.5 \\
        --sentinel <path>/dataset.json \\
        --summary <path>/summary.json
"""

import argparse
import json
import logging
from pathlib import Path

from data_quality_frame_level.dataset import iter_frames
from data_quality_frame_level.fiftyone_build import build_dataset
from data_quality_frame_level.inference import PredBBox

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--iou-thresh", required=True, type=float)
    parser.add_argument("--review-conf-thresh", required=True, type=float)
    parser.add_argument("--sentinel", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def _load_predictions(path: Path) -> dict[str, list[PredBBox]]:
    payload = json.loads(path.read_text())
    preds_by_stem: dict[str, list[PredBBox]] = {}
    for stem, entry in payload["frames"].items():
        preds_by_stem[stem] = [PredBBox(**p) for p in entry["predictions"]]
    return preds_by_stem


def main() -> None:
    args = _parse_args()
    args.sentinel.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    frames = list(iter_frames(args.split_dir))
    predictions = _load_predictions(args.predictions)

    logger.info(
        "Building FiftyOne dataset %s (frames=%d, frames_with_preds=%d)",
        args.dataset_name,
        len(frames),
        len(predictions),
    )
    _dataset, summary = build_dataset(
        dataset_name=args.dataset_name,
        frames=frames,
        predictions_by_stem=predictions,
        iou_thresh=args.iou_thresh,
        review_conf_thresh=args.review_conf_thresh,
    )

    args.sentinel.write_text(
        json.dumps(
            {"dataset_name": args.dataset_name, "num_samples": summary["num_samples"]},
            indent=2,
        )
    )
    args.summary.write_text(json.dumps(summary, indent=2))
    logger.info(
        "TP=%d FP=%d FN=%d precision=%.3f recall=%.3f",
        summary["tp"],
        summary["fp"],
        summary["fn"],
        summary["precision"],
        summary["recall"],
    )


if __name__ == "__main__":
    main()

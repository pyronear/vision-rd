"""Build a FiftyOne dataset from tracking pipeline outputs.

Creates a FiftyOne dataset where each frame is a sample with YOLO detections,
ground truth boxes, and sequence-level classification metadata (TP/FP/FN/TN).

Usage:
    uv run --group explore python scripts/build_fiftyone_dataset.py \
        --split val \
        --track-dir data/07_model_output/val \
        --infer-dir data/03_primary/val \
        --data-dir data/01_raw/datasets/val
"""

import argparse
import logging
from pathlib import Path

import fiftyone as fo
from tqdm import tqdm

from tracking_fsm_baseline.data import get_sorted_frames, load_label_boxes
from tracking_fsm_baseline.detector import load_inference_results
from tracking_fsm_baseline.evaluator import load_tracking_results
from tracking_fsm_baseline.types import Detection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def detection_to_fo(det: Detection, label: str = "smoke") -> fo.Detection:
    """Convert a pipeline Detection to a FiftyOne Detection.

    Transforms YOLO center-based (cx, cy, w, h) to FiftyOne top-left (x, y, w, h).
    """
    return fo.Detection(
        label=label,
        bounding_box=[
            det.cx - det.w / 2,
            det.cy - det.h / 2,
            det.w,
            det.h,
        ],
        confidence=det.confidence,
    )


def classify_sequence(is_positive_gt: bool, is_positive_pred: bool) -> str:
    """Return the TP/FP/FN/TN category string for a sequence."""
    if is_positive_gt and is_positive_pred:
        return "true_positive"
    if not is_positive_gt and is_positive_pred:
        return "false_positive"
    if is_positive_gt and not is_positive_pred:
        return "false_negative"
    return "true_negative"


def build_dataset(
    split: str,
    track_dir: Path,
    infer_dir: Path,
    data_dir: Path,
    dataset_name: str,
) -> fo.Dataset:
    """Build a FiftyOne dataset from tracking pipeline outputs."""
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
        logger.info("Deleted existing dataset '%s'.", dataset_name)

    # Load tracking results
    results = load_tracking_results(track_dir / "tracking_results.json")
    logger.info("Loaded %d tracking results.", len(results))

    # Build samples
    samples = []
    for result in tqdm(results, desc=f"Building {split} dataset"):
        seq_id = result["sequence_id"]
        category = classify_sequence(
            result["is_positive_gt"], result["is_positive_pred"]
        )

        # Load inference results
        infer_path = infer_dir / f"{seq_id}.json"
        if not infer_path.is_file():
            logger.warning("Missing inference file for %s, skipping.", seq_id)
            continue

        infer_frames = load_inference_results(infer_path)

        # Get sorted image paths
        seq_dir = data_dir / seq_id
        if not seq_dir.is_dir():
            logger.warning("Missing sequence directory for %s, skipping.", seq_id)
            continue

        image_paths = get_sorted_frames(seq_dir)
        if not image_paths:
            logger.warning("No images found for %s, skipping.", seq_id)
            continue

        # Map frame_id -> image path (for padded frames, reuse last image)
        image_by_frame_id = {p.stem: p for p in image_paths}
        last_img_path = image_paths[-1]

        for idx, infer_frame in enumerate(infer_frames):
            frame_id = infer_frame.frame_id
            img_path = image_by_frame_id.get(frame_id, last_img_path)

            # Load label boxes
            label_path = img_path.parent.parent / "labels" / (frame_id + ".txt")
            boxes, is_human = load_label_boxes(label_path)
            gt_boxes = boxes if is_human else []
            prior_boxes = boxes if not is_human else []

            # Build FiftyOne sample
            sample = fo.Sample(filepath=str(img_path.resolve()))
            sample.tags = [category]
            sample["sequence_id"] = seq_id
            sample["split"] = split
            sample["frame_index"] = idx
            sample["category"] = category
            sample["ground_truth"] = fo.Classification(
                label="smoke" if result["is_positive_gt"] else "no_smoke"
            )
            sample["prediction"] = fo.Classification(
                label="smoke" if result["is_positive_pred"] else "no_smoke"
            )
            sample["num_frames"] = result["num_frames"]
            sample["num_detections_total"] = result["num_detections_total"]
            sample["num_tracks"] = result["num_tracks"]
            sample["confirmed_frame_index"] = result["confirmed_frame_index"]

            # YOLO detections from current model
            if infer_frame.detections:
                sample["yolo_detections"] = fo.Detections(
                    detections=[detection_to_fo(d) for d in infer_frame.detections]
                )

            # Ground truth boxes (human annotations, green)
            if gt_boxes:
                sample["gt_detections"] = fo.Detections(
                    detections=[detection_to_fo(d, label="smoke_gt") for d in gt_boxes]
                )

            # Prior YOLO predictions from labels (purple)
            if prior_boxes:
                sample["prior_detections"] = fo.Detections(
                    detections=[
                        detection_to_fo(d, label="smoke_prior") for d in prior_boxes
                    ]
                )

            samples.append(sample)

    # Create dataset and add samples
    dataset = fo.Dataset(name=dataset_name, persistent=True)
    dataset.add_samples(samples)

    # Apply color scheme matching src/visualization.py colors
    # GT_COLOR=(0,180,0) PRIOR_COLOR=(140,0,200) PRED_COLOR=(220,0,0)
    dataset.app_config.color_scheme = fo.ColorScheme(
        fields=[
            {"path": "gt_detections", "fieldColor": "#00B400"},
            {"path": "prior_detections", "fieldColor": "#8C00C8"},
            {"path": "yolo_detections", "fieldColor": "#DC0000"},
        ]
    )
    dataset.save()

    logger.info(
        "Created dataset '%s' with %d samples from %d sequences.",
        dataset_name,
        len(samples),
        len(results),
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FiftyOne dataset from tracking pipeline outputs."
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Dataset split (train or val).",
    )
    parser.add_argument(
        "--track-dir",
        type=Path,
        required=True,
        help="Path to tracking results directory.",
    )
    parser.add_argument(
        "--infer-dir",
        type=Path,
        required=True,
        help="Path to padded inference results directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to raw dataset directory (images and labels).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="FiftyOne dataset name (default: tracking-fsm-{split}).",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or f"tracking-fsm-{args.split}"
    build_dataset(
        split=args.split,
        track_dir=args.track_dir,
        infer_dir=args.infer_dir,
        data_dir=args.data_dir,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()

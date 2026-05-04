"""Build fire-tubes from padded inference results and original images.

For each sequence, tracks detections across frames via IoU matching,
crops the corresponding image regions, and saves the tubes to disk.

Usage:
    uv run python scripts/build_tubes.py \
        --infer-dir data/03_primary/train \
        --data-dir data/01_raw/datasets/train \
        --output-dir data/04_feature/train \
        --crop-size 64 \
        --max-tube-length 50 \
        --confidence-threshold 0.3 \
        --max-detection-area 0.05 \
        --iou-threshold 0.1
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data import is_wf_sequence
from src.detector import load_inference_results
from src.tube import build_tubes_for_sequence

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fire-tubes from detections.")
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, required=True)
    parser.add_argument("--max-tube-length", type=int, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--max-detection-area", type=float, required=True)
    parser.add_argument("--iou-threshold", type=float, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Found %d inference files.", len(infer_files))

    total_tubes = 0
    for infer_path in tqdm(infer_files, desc="Building tubes"):
        seq_id = infer_path.stem
        seq_output_dir = args.output_dir / seq_id
        if seq_output_dir.exists():
            continue

        frame_results = load_inference_results(infer_path)
        image_dir = args.data_dir / seq_id / "images"

        tubes = build_tubes_for_sequence(
            frame_results=frame_results,
            image_dir=image_dir,
            sequence_id=seq_id,
            crop_size=args.crop_size,
            max_tube_length=args.max_tube_length,
            confidence_threshold=args.confidence_threshold,
            max_detection_area=args.max_detection_area,
            iou_threshold=args.iou_threshold,
        )

        # Assign ground-truth labels (sequence-level)
        is_positive = is_wf_sequence(args.data_dir / seq_id)
        for tube in tubes:
            tube.label = is_positive

        # Save tubes
        seq_output_dir.mkdir(parents=True, exist_ok=True)
        _save_tubes(tubes, seq_output_dir)
        total_tubes += len(tubes)

    logger.info(
        "Built %d tubes across %d sequences. Saved to %s",
        total_tubes,
        len(infer_files),
        args.output_dir,
    )


def _save_tubes(tubes: list, output_dir: Path) -> None:
    """Save tubes as .npz (crops) + .json (metadata) per sequence."""
    if not tubes:
        # Save empty metadata so downstream stages know this was processed
        (output_dir / "metadata.json").write_text(json.dumps([]))
        return

    all_crops = []
    metadata = []
    for tube in tubes:
        tube_meta = {
            "tube_id": tube.tube_id,
            "sequence_id": tube.sequence_id,
            "label": tube.label,
            "crops": [],
        }
        for crop in tube.crops:
            crop_idx = len(all_crops)
            all_crops.append(crop.image)
            tube_meta["crops"].append(
                {
                    "crop_idx": crop_idx,
                    "frame_id": crop.frame_id,
                    "timestamp": crop.timestamp.isoformat(),
                    "detection": {
                        "class_id": crop.detection.class_id,
                        "cx": crop.detection.cx,
                        "cy": crop.detection.cy,
                        "w": crop.detection.w,
                        "h": crop.detection.h,
                        "confidence": crop.detection.confidence,
                    },
                }
            )
        metadata.append(tube_meta)

    # Save crops as a single numpy array
    crops_array = np.stack(all_crops)  # shape (N, crop_size, crop_size, 3)
    np.savez_compressed(output_dir / "crops.npz", crops=crops_array)

    # Save metadata
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

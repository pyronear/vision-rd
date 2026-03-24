"""Generate annotated frame strips for mispredicted sequences (FP/FN).

Loads tracking results to identify mispredictions, then for each one
renders a horizontal strip of annotated frames showing YOLO detections
(red) and ground truth boxes (green). Outputs are saved to separate
false_positives/ and false_negatives/ subdirectories.

Usage:
    uv run python scripts/visualize_mispredictions.py \
        --track-dir data/07_model_output/val \
        --infer-dir data/02_intermediate/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/08_reporting/val/mispredictions
"""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from src.data import get_sorted_frames, parse_timestamp
from src.detector import load_inference_results
from src.evaluator import load_tracking_results
from src.visualization import load_gt_boxes, render_sequence_strip

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualizations for mispredicted sequences."
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
        help="Path to inference results directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to raw dataset directory (images and labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for misprediction visualizations.",
    )
    args = parser.parse_args()

    fp_dir = args.output_dir / "false_positives"
    fn_dir = args.output_dir / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    # Load tracking results and filter mispredictions
    results = load_tracking_results(args.track_dir / "tracking_results.json")

    false_positives = [
        r for r in results if not r["is_positive_gt"] and r["is_positive_pred"]
    ]
    false_negatives = [
        r for r in results if r["is_positive_gt"] and not r["is_positive_pred"]
    ]

    logger.info(
        "Found %d false positives and %d false negatives.",
        len(false_positives),
        len(false_negatives),
    )

    # Process each misprediction
    for error_type, sequences, out_dir in [
        ("false_positive", false_positives, fp_dir),
        ("false_negative", false_negatives, fn_dir),
    ]:
        for result in tqdm(sequences, desc=error_type.replace("_", " ").title()):
            seq_id = result["sequence_id"]

            # Load inference results
            infer_path = args.infer_dir / f"{seq_id}.json"
            if not infer_path.is_file():
                logger.warning("Missing inference file for %s, skipping.", seq_id)
                continue

            infer_frames = load_inference_results(infer_path)

            # Load raw image paths
            seq_dir = args.data_dir / seq_id
            if not seq_dir.is_dir():
                logger.warning("Missing sequence directory for %s, skipping.", seq_id)
                continue

            image_paths = get_sorted_frames(seq_dir)
            if not image_paths:
                logger.warning("No images found for %s, skipping.", seq_id)
                continue

            # Build frame-level data by matching images to inference results
            infer_by_frame_id = {f.frame_id: f for f in infer_frames}
            frames_data = []

            for idx, img_path in enumerate(image_paths):
                frame_id = img_path.stem
                timestamp = parse_timestamp(img_path.name)
                timestamp_str = timestamp.strftime("%H:%M:%S")

                # Load GT boxes from label file
                label_path = img_path.parent.parent / "labels" / (frame_id + ".txt")
                gt_boxes = load_gt_boxes(label_path)

                # Get prediction detections from inference
                pred_detections = []
                if frame_id in infer_by_frame_id:
                    pred_detections = infer_by_frame_id[frame_id].detections

                frames_data.append(
                    {
                        "image_path": img_path,
                        "gt_boxes": gt_boxes,
                        "pred_detections": pred_detections,
                        "frame_index": idx,
                        "timestamp_str": timestamp_str,
                        "num_preds": len(pred_detections),
                    }
                )

            metadata = {
                "sequence_id": seq_id,
                "error_type": error_type,
                "is_positive_gt": result["is_positive_gt"],
                "is_positive_pred": result["is_positive_pred"],
                "num_frames": result["num_frames"],
                "num_detections_total": result["num_detections_total"],
                "num_tracks": result["num_tracks"],
                "confirmed_frame_index": result["confirmed_frame_index"],
                "confirmed_timestamp": result["confirmed_timestamp"],
            }

            output_path = out_dir / f"{seq_id}.png"
            render_sequence_strip(frames_data, metadata, output_path)

    # Write summary
    summary = {
        "num_false_positives": len(false_positives),
        "num_false_negatives": len(false_negatives),
        "false_positive_sequences": [r["sequence_id"] for r in false_positives],
        "false_negative_sequences": [r["sequence_id"] for r in false_negatives],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary to %s", summary_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()

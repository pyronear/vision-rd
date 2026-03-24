"""Generate annotated frame strips for tracking predictions.

Loads tracking results and renders annotated frame strips showing YOLO
detections (red) and ground truth boxes (green). All false positives and
false negatives are generated; true positives and true negatives are
randomly sampled.

Usage:
    uv run python scripts/visualize_mispredictions.py \
        --track-dir data/07_model_output/val \
        --infer-dir data/02_intermediate/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/08_reporting/val/mispredictions \
        --max-samples 20 \
        --seed 42
"""

import argparse
import json
import logging
import random
from pathlib import Path

from tqdm import tqdm

from src.data import get_sorted_frames, parse_timestamp
from src.detector import load_inference_results
from src.evaluator import load_tracking_results
from src.visualization import load_label_boxes, render_sequence_strip

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def render_sequences(
    sequences: list[dict],
    category: str,
    out_dir: Path,
    infer_dir: Path,
    data_dir: Path,
) -> None:
    """Render frame strips for a list of sequences."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in tqdm(sequences, desc=category.replace("_", " ").title()):
        seq_id = result["sequence_id"]

        infer_path = infer_dir / f"{seq_id}.json"
        if not infer_path.is_file():
            logger.warning("Missing inference file for %s, skipping.", seq_id)
            continue

        infer_frames = load_inference_results(infer_path)

        seq_dir = data_dir / seq_id
        if not seq_dir.is_dir():
            logger.warning("Missing sequence directory for %s, skipping.", seq_id)
            continue

        image_paths = get_sorted_frames(seq_dir)
        if not image_paths:
            logger.warning("No images found for %s, skipping.", seq_id)
            continue

        infer_by_frame_id = {f.frame_id: f for f in infer_frames}
        frames_data = []

        for idx, img_path in enumerate(image_paths):
            frame_id = img_path.stem
            timestamp = parse_timestamp(img_path.name)
            timestamp_str = timestamp.strftime("%H:%M:%S")

            label_path = img_path.parent.parent / "labels" / (frame_id + ".txt")
            boxes, is_human = load_label_boxes(label_path)
            gt_boxes = boxes if is_human else []
            prior_boxes = boxes if not is_human else []

            pred_detections = []
            if frame_id in infer_by_frame_id:
                pred_detections = infer_by_frame_id[frame_id].detections

            frames_data.append(
                {
                    "image_path": img_path,
                    "gt_boxes": gt_boxes,
                    "prior_boxes": prior_boxes,
                    "pred_detections": pred_detections,
                    "frame_index": idx,
                    "timestamp_str": timestamp_str,
                    "num_preds": len(pred_detections),
                }
            )

        metadata = {
            "sequence_id": seq_id,
            "error_type": category,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualizations for tracking predictions."
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
        help="Output directory for visualizations.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max number of TP/TN sequences to visualize (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducible TP/TN sampling.",
    )
    args = parser.parse_args()

    # Load tracking results and split into categories
    results = load_tracking_results(args.track_dir / "tracking_results.json")

    categories = {
        "true_positive": [
            r for r in results if r["is_positive_gt"] and r["is_positive_pred"]
        ],
        "true_negative": [
            r for r in results if not r["is_positive_gt"] and not r["is_positive_pred"]
        ],
        "false_positive": [
            r for r in results if not r["is_positive_gt"] and r["is_positive_pred"]
        ],
        "false_negative": [
            r for r in results if r["is_positive_gt"] and not r["is_positive_pred"]
        ],
    }

    for cat, seqs in categories.items():
        logger.info("  %s: %d", cat, len(seqs))

    # Sample TP/TN, keep all FP/FN
    rng = random.Random(args.seed)
    sampled = {}
    for cat, seqs in categories.items():
        if cat in ("true_positive", "true_negative"):
            n = min(args.max_samples, len(seqs))
            sampled[cat] = rng.sample(seqs, n)
            logger.info("Sampled %d/%d %s sequences.", n, len(seqs), cat)
        else:
            sampled[cat] = seqs

    # Render each category
    for cat, seqs in sampled.items():
        out_dir = args.output_dir / f"{cat}s"
        render_sequences(seqs, cat, out_dir, args.infer_dir, args.data_dir)

    # Write summary
    summary = {}
    for cat, seqs in categories.items():
        summary[f"num_{cat}s"] = len(seqs)
    for cat, seqs in sampled.items():
        summary[f"{cat}_sequences_visualized"] = [r["sequence_id"] for r in seqs]
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary to %s", summary_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()

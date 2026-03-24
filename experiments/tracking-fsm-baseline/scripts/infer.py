"""Run YOLO inference on all sequences in a data split.

Iterates over sequence directories, runs the YOLO model on each frame,
and writes one JSON file per sequence containing frame-level detections
(bounding boxes and confidence scores). Already-processed sequences are
skipped automatically.

Usage:
    uv run python scripts/infer.py \
        --data-dir data/01_raw/datasets/val \
        --model-path data/01_raw/models/best.pt \
        --output-dir data/02_intermediate/val \
        --confidence-threshold 0.1 \
        --iou-nms 0.5 \
        --image-size 1024
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from src.data import list_sequences
from src.detector import load_model, run_inference_on_sequence, save_inference_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on sequences.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to sequence split directory (e.g. data/01_raw/val).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to YOLO model weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for inference JSON files.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        required=True,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--iou-nms",
        type=float,
        required=True,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        required=True,
        help="YOLO input image size.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)

    sequences = list_sequences(args.data_dir)
    logger.info("Found %d sequences in %s", len(sequences), args.data_dir)

    for seq_dir in tqdm(sequences, desc="Inference"):
        output_path = args.output_dir / f"{seq_dir.name}.json"
        if output_path.exists():
            continue

        results = run_inference_on_sequence(
            model=model,
            sequence_dir=seq_dir,
            conf=args.confidence_threshold,
            iou_nms=args.iou_nms,
            img_size=args.image_size,
        )
        save_inference_results(results, output_path)

    logger.info("Inference complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()

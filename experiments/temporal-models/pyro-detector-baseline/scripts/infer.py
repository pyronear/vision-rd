"""Run YOLO classifier on all frames and cache per-frame detections.

Saves post-processed bounding-box predictions as one JSON per sequence,
enabling fast parameter sweeps without re-running model inference.

Usage:
    uv run python scripts/infer.py \
        --data-dir data/01_raw/datasets/val \
        --model-dir data/01_raw/models \
        --output-dir data/03_primary/val \
        --model-conf-thresh 0.05 \
        --max-bbox-size 0.4
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image
from pyro_predictor.vision import Classifier
from tqdm import tqdm

from pyro_detector_baseline.data import get_sorted_frames, list_sequences

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _find_onnx_model(model_dir: Path) -> str | None:
    """Find the ONNX model file in a directory, skipping macOS resource forks."""
    onnx_files = [f for f in model_dir.glob("**/*.onnx") if not f.name.startswith("._")]
    return str(onnx_files[0]) if onnx_files else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO classifier on sequence frames."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to sequence data directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to directory containing the ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for per-sequence detection JSONs.",
    )
    parser.add_argument(
        "--model-conf-thresh",
        type=float,
        default=0.05,
        help="Per-frame YOLO confidence threshold.",
    )
    parser.add_argument(
        "--max-bbox-size",
        type=float,
        default=0.4,
        help="Maximum detection width as image fraction.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = _find_onnx_model(args.model_dir)
    logger.info("Using model: %s", model_path)

    classifier = Classifier(
        model_path=model_path,
        conf=args.model_conf_thresh,
        max_bbox_size=args.max_bbox_size,
        verbose=False,
    )

    sequences = list_sequences(args.data_dir)
    logger.info("Found %d sequences.", len(sequences))

    total_frames = 0
    for seq_dir in tqdm(sequences, desc="Inferring"):
        seq_id = seq_dir.name
        frame_paths = get_sorted_frames(seq_dir)

        if not frame_paths:
            logger.warning("No frames in %s, skipping.", seq_id)
            continue

        frames_data: list[dict] = []
        for frame_path in frame_paths:
            pil_img = Image.open(frame_path).convert("RGB")
            preds = classifier(pil_img)
            frames_data.append(
                {
                    "filename": frame_path.name,
                    "detections": preds.tolist(),
                }
            )
            total_frames += 1

        output_path = args.output_dir / f"{seq_id}.json"
        output_path.write_text(json.dumps(frames_data))

    logger.info(
        "Saved detections for %d sequences (%d frames) to %s",
        len(sequences),
        total_frames,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

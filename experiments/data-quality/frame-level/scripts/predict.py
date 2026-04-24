"""Run YOLO on every frame of one split; write predictions.json.

Usage::

    uv run python scripts/predict.py \\
        --model-name yolo11s-nimble-narwhal \\
        --model-path data/01_raw/models/yolo11s-nimble-narwhal.pt \\
        --split-dir data/01_raw/datasets/train \\
        --conf-thresh 0.35 \\
        --output-dir data/07_model_output/yolo11s-nimble-narwhal/train
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from data_quality_frame_level.dataset import iter_frames
from data_quality_frame_level.inference import load_model, predict_images

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--conf-thresh", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images per inference call.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="ultralytics device string (default: first CUDA GPU).",
    )
    return parser.parse_args()


def _chunks(seq: list, size: int):
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = list(iter_frames(args.split_dir))
    logger.info("Discovered %d frames in %s", len(frames), args.split_dir)

    model = load_model(args.model_path)

    frames_out: dict[str, dict] = {}
    for batch in tqdm(list(_chunks(frames, args.batch_size)), desc="YOLO inference"):
        preds_batch = predict_images(
            model,
            [f.image_path for f in batch],
            conf_thresh=args.conf_thresh,
            device=args.device,
        )
        for frame, preds in zip(batch, preds_batch, strict=True):
            frames_out[frame.stem] = {
                "image_path": str(frame.image_path.relative_to(args.split_dir)),
                "predictions": [asdict(p) for p in preds],
            }

    payload = {
        "model_name": args.model_name,
        "split_dir": str(args.split_dir),
        "conf_thresh": args.conf_thresh,
        "frames": frames_out,
    }
    (args.output_dir / "predictions.json").write_text(json.dumps(payload))
    logger.info(
        "Wrote %d frame predictions to %s",
        len(frames_out),
        args.output_dir / "predictions.json",
    )


if __name__ == "__main__":
    main()

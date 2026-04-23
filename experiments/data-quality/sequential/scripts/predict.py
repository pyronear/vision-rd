"""Run a TemporalModel on every sequence of one split, dump predictions.

Usage::

    uv run python scripts/predict.py \\
        --model-name bbox-tube-temporal-vit-dinov2-finetune \\
        --model-type bbox-tube-temporal \\
        --model-package data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip \\
        --split-dir data/01_raw/datasets/val \\
        --split val \\
        --output-dir data/07_model_output/bbox-tube-temporal-vit-dinov2-finetune/val
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from tqdm import tqdm

from data_quality_sequential.dataset import iter_sequences
from data_quality_sequential.registry import MODEL_REGISTRY, load_model
from data_quality_sequential.review import Prediction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-type", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--model-package", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s from %s", args.model_type, args.model_package)
    model = load_model(args.model_type, args.model_package)

    refs = list(iter_sequences(args.split_dir, split=args.split))
    logger.info("Found %d sequences in %s", len(refs), args.split_dir)

    predictions: list[Prediction] = []
    for ref in tqdm(refs, desc=f"predict[{args.split}]", unit="seq"):
        output = model.predict_sequence(ref.frame_paths)
        predictions.append(
            Prediction(
                sequence_name=ref.name,
                predicted=output.is_positive,
                trigger_frame_index=output.trigger_frame_index,
            )
        )

    preds_path = args.output_dir / "predictions.json"
    preds_path.write_text(
        json.dumps([dataclasses.asdict(p) for p in predictions], indent=2)
    )
    logger.info("Wrote %d predictions to %s", len(predictions), preds_path)


if __name__ == "__main__":
    main()

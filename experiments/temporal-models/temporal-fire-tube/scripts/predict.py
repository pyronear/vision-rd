"""Run fire-tube predictions on a data split and produce sequence-level results.

Loads the trained classifier, feature files, and ground-truth labels,
then classifies each tube and aggregates to sequence-level predictions.

Usage:
    uv run python scripts/predict.py \
        --feature-dir data/05_model_input/val \
        --model-dir data/06_models \
        --data-dir data/01_raw/datasets/val \
        --infer-dir data/03_primary/val \
        --output-dir data/07_model_output/val
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.classifier import load_model, predict_tubes
from src.data import is_wf_sequence
from src.detector import load_inference_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict on a split using trained fire-tube classifier."
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load classifier
    clf = load_model(args.model_dir / "classifier.pkl")
    logger.info("Loaded classifier from %s", args.model_dir)

    # Process each sequence
    feature_files = sorted(args.feature_dir.glob("*.npz"))
    logger.info("Processing %d sequences.", len(feature_files))

    results = []
    for fpath in feature_files:
        seq_id = fpath.stem
        data = np.load(fpath)
        features = data["features"]

        # Ground truth
        is_positive_gt = is_wf_sequence(args.data_dir / seq_id)

        # Load inference results for metadata
        infer_path = args.infer_dir / f"{seq_id}.json"
        frame_results = (
            load_inference_results(infer_path) if infer_path.exists() else []
        )
        num_frames = len(frame_results)
        num_detections_total = sum(len(f.detections) for f in frame_results)
        first_timestamp = (
            frame_results[0].timestamp.isoformat() if frame_results else None
        )

        # Predict
        is_positive_pred = False
        confirmed_timestamp = None
        confirmed_frame_index = None
        num_tubes = features.shape[0]

        if num_tubes > 0:
            predictions, _confidences = predict_tubes(features, clf)
            if predictions.any():
                is_positive_pred = True
                # Use the first positive tube's info for trigger frame
                # We don't have per-tube frame info here, so use first_timestamp
                # as a conservative estimate
                confirmed_timestamp = first_timestamp
                confirmed_frame_index = 0

        results.append(
            {
                "sequence_id": seq_id,
                "is_positive_gt": is_positive_gt,
                "is_positive_pred": is_positive_pred,
                "num_frames": num_frames,
                "num_detections_total": num_detections_total,
                "num_tubes": num_tubes,
                "confirmed_frame_index": confirmed_frame_index,
                "confirmed_timestamp": confirmed_timestamp,
                "first_timestamp": first_timestamp,
            }
        )

    # Save results
    output_path = args.output_dir / "prediction_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved %d prediction results to %s", len(results), output_path)

    # Quick summary
    tp = sum(1 for r in results if r["is_positive_gt"] and r["is_positive_pred"])
    fp = sum(1 for r in results if not r["is_positive_gt"] and r["is_positive_pred"])
    fn = sum(1 for r in results if r["is_positive_gt"] and not r["is_positive_pred"])
    tn = sum(
        1 for r in results if not r["is_positive_gt"] and not r["is_positive_pred"]
    )
    logger.info("TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)


if __name__ == "__main__":
    main()

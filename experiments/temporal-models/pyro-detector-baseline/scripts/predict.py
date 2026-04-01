"""Run pyro-predictor on all sequences in a data split.

Iterates over sequence directories, feeds each frame through the
Predictor, and writes sequence-level results to tracking_results.json.
The output format matches the FSM tracking baseline for evaluation
compatibility.

Usage:
    uv run python scripts/predict.py \
        --data-dir data/01_raw/datasets/val \
        --model-dir data/01_raw/models \
        --output-dir data/07_model_output/val \
        --conf-thresh 0.35 \
        --model-conf-thresh 0.05 \
        --nb-consecutive-frames 7 \
        --max-bbox-size 0.4
"""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from pyro_detector_baseline.data import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
    parse_timestamp,
)
from pyro_detector_baseline.predictor_wrapper import (
    create_predictor,
    predict_sequence,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pyro-predictor on sequence data.")
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
        help="Output directory for prediction results.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.15,
        help="Predictor confidence threshold for alerts.",
    )
    parser.add_argument(
        "--model-conf-thresh",
        type=float,
        default=0.05,
        help="Per-frame YOLO confidence threshold.",
    )
    parser.add_argument(
        "--nb-consecutive-frames",
        type=int,
        default=8,
        help="Temporal sliding window size.",
    )
    parser.add_argument(
        "--max-bbox-size",
        type=float,
        default=0.4,
        help="Maximum detection width as image fraction.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find ONNX model in model directory (skip macOS resource forks)
    onnx_files = [
        f for f in args.model_dir.glob("**/*.onnx") if not f.name.startswith("._")
    ]
    model_path = str(onnx_files[0]) if onnx_files else None
    logger.info("Using model: %s", model_path)

    predictor = create_predictor(
        model_path=model_path,
        conf_thresh=args.conf_thresh,
        model_conf_thresh=args.model_conf_thresh,
        nb_consecutive_frames=args.nb_consecutive_frames,
        max_bbox_size=args.max_bbox_size,
    )

    sequences = list_sequences(args.data_dir)
    logger.info("Found %d sequences.", len(sequences))

    results: list[dict] = []
    for seq_dir in tqdm(sequences, desc="Predicting"):
        seq_id = seq_dir.name
        frame_paths = get_sorted_frames(seq_dir)

        if not frame_paths:
            logger.warning("No frames in %s, skipping.", seq_id)
            continue

        is_alarm, trigger_idx, confidences = predict_sequence(
            predictor=predictor,
            frame_paths=frame_paths,
            cam_id=seq_id,
        )

        gt = is_wf_sequence(seq_dir)
        first_ts = parse_timestamp(frame_paths[0].name)
        confirmed_ts = (
            parse_timestamp(frame_paths[trigger_idx].name)
            if trigger_idx is not None
            else None
        )

        results.append(
            {
                "sequence_id": seq_id,
                "is_positive_gt": gt,
                "is_positive_pred": is_alarm,
                "num_frames": len(frame_paths),
                "num_detections_total": sum(1 for c in confidences if c > 0),
                "confirmed_frame_index": trigger_idx,
                "confirmed_timestamp": (
                    confirmed_ts.isoformat() if confirmed_ts else None
                ),
                "first_timestamp": first_ts.isoformat(),
                "per_frame_confidences": confidences,
            }
        )

    output_path = args.output_dir / "tracking_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved %d results to %s", len(results), output_path)

    # Quick summary
    tp = sum(1 for r in results if r["is_positive_gt"] and r["is_positive_pred"])
    fp = sum(1 for r in results if not r["is_positive_gt"] and r["is_positive_pred"])
    fn = sum(1 for r in results if r["is_positive_gt"] and not r["is_positive_pred"])
    tn = sum(
        1 for r in results if not r["is_positive_gt"] and not r["is_positive_pred"]
    )
    logger.info("  TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)


if __name__ == "__main__":
    main()

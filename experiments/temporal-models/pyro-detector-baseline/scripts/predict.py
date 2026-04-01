"""Replay temporal logic on cached YOLO detections for a data split.

Loads per-frame detections produced by the infer stage and replays them
through the Predictor's sliding-window temporal logic.  Writes
sequence-level results to tracking_results.json (same format consumed
by the evaluate stage).

Usage:
    uv run python scripts/predict.py \
        --infer-dir data/03_primary/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/07_model_output/val \
        --conf-thresh 0.35 \
        --nb-consecutive-frames 7
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from pyro_predictor.predictor import Predictor
from tqdm import tqdm

from pyro_detector_baseline.data import is_wf_sequence, parse_timestamp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DUMMY_FRAME = Image.new("RGB", (1, 1))


def _create_replay(conf_thresh: float, nb_consecutive_frames: int) -> Predictor:
    """Create a lightweight Predictor for temporal replay without loading YOLO."""
    replay = object.__new__(Predictor)
    replay.conf_thresh = conf_thresh
    replay.nb_consecutive_frames = nb_consecutive_frames
    replay._states = {}
    return replay


def _load_sequence_detections(
    json_path: Path,
) -> list[tuple[str, np.ndarray]]:
    """Load cached per-frame detections from an infer JSON."""
    data = json.loads(json_path.read_text())
    return [
        (d["filename"], np.array(d["detections"], dtype=np.float64).reshape(-1, 5))
        for d in data
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay temporal logic on cached detections."
    )
    parser.add_argument(
        "--infer-dir",
        type=Path,
        required=True,
        help="Path to cached inference results directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to sequence data directory (for GT labels).",
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
        default=0.35,
        help="Predictor confidence threshold for alerts.",
    )
    parser.add_argument(
        "--nb-consecutive-frames",
        type=int,
        default=7,
        help="Temporal sliding window size.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    replay = _create_replay(args.conf_thresh, args.nb_consecutive_frames)

    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Found %d inference files.", len(infer_files))

    results: list[dict] = []
    for infer_path in tqdm(infer_files, desc="Predicting"):
        seq_id = infer_path.stem
        frame_detections = _load_sequence_detections(infer_path)

        if not frame_detections:
            logger.warning("No frames in %s, skipping.", seq_id)
            continue

        # Locate sequence dir for GT (handles nested wildfire/fp layout)
        seq_dir = args.data_dir / seq_id
        if not seq_dir.is_dir():
            candidates = list(args.data_dir.glob(f"*/{seq_id}"))
            if not candidates:
                logger.warning("No data dir for %s, skipping.", seq_id)
                continue
            seq_dir = candidates[0]

        gt = is_wf_sequence(seq_dir)
        first_ts = parse_timestamp(frame_detections[0][0])

        # Replay temporal logic
        cam_key = seq_id
        replay._states[cam_key] = replay._new_state()

        confidences: list[float] = []
        trigger_idx: int | None = None
        for i, (_filename, preds) in enumerate(frame_detections):
            conf = replay._update_states(_DUMMY_FRAME, preds, cam_key)
            confidences.append(float(conf))
            if conf > args.conf_thresh and trigger_idx is None:
                trigger_idx = i

        del replay._states[cam_key]

        confirmed_ts = (
            parse_timestamp(frame_detections[trigger_idx][0])
            if trigger_idx is not None
            else None
        )

        results.append(
            {
                "sequence_id": seq_id,
                "is_positive_gt": gt,
                "is_positive_pred": trigger_idx is not None,
                "num_frames": len(frame_detections),
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

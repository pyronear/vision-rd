"""Apply FSM-based temporal tracking to YOLO inference results.

Loads per-sequence inference JSONs, filters detections by confidence,
and runs the SimpleTracker (IoU matching + min-consecutive-frames FSM)
to decide whether each sequence triggers an alarm. Results are joined
with ground-truth labels and written to a single tracking_results.json.

Usage:
    uv run python scripts/track.py \
        --infer-dir data/02_intermediate/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/07_model_output/val \
        --confidence-threshold 0.25 \
        --iou-threshold 0.3 \
        --min-consecutive 3
"""

import argparse
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.data import is_wf_sequence
from src.detector import load_inference_results
from src.tracker import SimpleTracker
from src.types import SequenceResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply tracking to inference results.")
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
        help="Path to sequence data directory (for GT labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for tracking results.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        required=True,
        help="Min confidence to keep a detection for tracking.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        required=True,
        help="IoU threshold for detection matching.",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        required=True,
        help="Min consecutive frames for confirmation.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracker = SimpleTracker(
        iou_threshold=args.iou_threshold,
        min_consecutive=args.min_consecutive,
    )

    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Found %d inference result files.", len(infer_files))

    results: list[dict] = []
    for infer_path in tqdm(infer_files, desc="Tracking"):
        seq_id = infer_path.stem
        frames = load_inference_results(infer_path)

        # Filter detections by confidence threshold
        for frame in frames:
            frame.detections = [
                d for d in frame.detections if d.confidence >= args.confidence_threshold
            ]

        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        gt = is_wf_sequence(args.data_dir / seq_id)

        total_dets = sum(len(f.detections) for f in frames)
        first_ts = frames[0].timestamp if frames else None
        confirmed_ts = (
            frames[confirmed_idx].timestamp if confirmed_idx is not None else None
        )

        seq_result = SequenceResult(
            sequence_id=seq_id,
            is_positive_gt=gt,
            is_positive_pred=is_alarm,
            num_frames=len(frames),
            num_detections_total=total_dets,
            num_tracks=len(tracks),
            confirmed_frame_index=confirmed_idx,
            confirmed_timestamp=confirmed_ts,
            first_timestamp=first_ts,
        )

        row = dataclasses.asdict(seq_result)
        for key in ("confirmed_timestamp", "first_timestamp"):
            val = row[key]
            row[key] = val.isoformat() if isinstance(val, datetime) else val
        results.append(row)

    output_path = args.output_dir / "tracking_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved %d tracking results to %s", len(results), output_path)

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

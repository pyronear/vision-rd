"""Apply tracking to YOLO inference results.

Loads per-sequence inference JSON, applies the tracker,
and saves results with ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from src.data import is_wf_sequence, load_wf_folders
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
        "--wf-registry",
        type=Path,
        required=True,
        help="Path to WF registry JSON.",
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

    wf_folders = load_wf_folders(args.wf_registry)
    logger.info("Loaded %d WF folder names from registry.", len(wf_folders))

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
                d
                for d in frame.detections
                if d.confidence >= args.confidence_threshold
            ]

        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        gt = is_wf_sequence(seq_id, wf_folders)

        total_dets = sum(len(f.detections) for f in frames)
        first_ts = frames[0].timestamp if frames else None
        confirmed_ts = (
            frames[confirmed_idx].timestamp
            if confirmed_idx is not None
            else None
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

        results.append(
            {
                "sequence_id": seq_result.sequence_id,
                "is_positive_gt": seq_result.is_positive_gt,
                "is_positive_pred": seq_result.is_positive_pred,
                "num_frames": seq_result.num_frames,
                "num_detections_total": seq_result.num_detections_total,
                "num_tracks": seq_result.num_tracks,
                "confirmed_frame_index": seq_result.confirmed_frame_index,
                "confirmed_timestamp": (
                    seq_result.confirmed_timestamp.isoformat()
                    if seq_result.confirmed_timestamp
                    else None
                ),
                "first_timestamp": (
                    seq_result.first_timestamp.isoformat()
                    if seq_result.first_timestamp
                    else None
                ),
            }
        )

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

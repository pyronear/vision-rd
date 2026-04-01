"""Apply MTB change detection + tracking to YOLO inference results.

Loads per-sequence inference JSONs, computes pixel-wise change masks
between consecutive frames, validates detections by change ratio,
and runs the SimpleTracker to decide whether each sequence triggers
an alarm. Results are joined with ground-truth labels and written to
a single tracking_results.json.

Usage:
    uv run python scripts/track.py \
        --infer-dir data/03_primary/val \
        --data-dir data/01_raw/datasets/val \
        --output-dir data/07_model_output/val \
        --pixel-threshold 19 \
        --min-change-ratio 0.01 \
        --confidence-threshold 0.01 \
        --iou-threshold 0.1 \
        --min-consecutive 2
"""

import argparse
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mtb_change_detection.change_detector import (
    compute_change_mask,
    compute_change_ratio_in_bbox,
)
from mtb_change_detection.data import (
    get_sorted_frames,
    is_wf_sequence,
)
from mtb_change_detection.detector import load_inference_results
from mtb_change_detection.tracker import SimpleTracker
from mtb_change_detection.types import FrameResult, SequenceResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _find_sequence_dir(data_dir: Path, seq_id: str) -> Path | None:
    """Find a sequence directory by ID within the nested layout."""
    for category in ("wildfire", "fp"):
        candidate = data_dir / category / seq_id
        if candidate.is_dir():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply MTB change detection + tracking."
    )
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # Change detection params
    parser.add_argument("--pixel-threshold", type=int, required=True)
    parser.add_argument("--min-change-ratio", type=float, required=True)
    # Pre-filter params
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--max-detection-area", type=float, default=None)
    # Tracker params
    parser.add_argument("--iou-threshold", type=float, required=True)
    parser.add_argument("--min-consecutive", type=int, required=True)
    parser.add_argument("--max-misses", type=int, default=0)
    parser.add_argument("--use-confidence-filter", type=str, default="false")
    parser.add_argument("--min-mean-confidence", type=float, default=0.0)
    parser.add_argument("--use-area-change-filter", type=str, default="false")
    parser.add_argument("--min-area-change", type=float, default=1.0)
    args = parser.parse_args()

    use_confidence_filter = args.use_confidence_filter.lower() == "true"
    use_area_change_filter = args.use_area_change_filter.lower() == "true"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracker = SimpleTracker(
        iou_threshold=args.iou_threshold,
        min_consecutive=args.min_consecutive,
        max_misses=args.max_misses,
        use_confidence_filter=use_confidence_filter,
        min_mean_confidence=args.min_mean_confidence,
        use_area_change_filter=use_area_change_filter,
        min_area_change=args.min_area_change,
    )

    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Found %d inference result files.", len(infer_files))

    results: list[dict] = []
    for infer_path in tqdm(infer_files, desc="MTB+Tracking"):
        seq_id = infer_path.stem
        frames = load_inference_results(infer_path)

        # Pre-filter detections
        max_area = args.max_detection_area
        frames = [
            FrameResult(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=[
                    d
                    for d in frame.detections
                    if d.confidence >= args.confidence_threshold
                    and (max_area is None or d.w * d.h <= max_area)
                ],
            )
            for frame in frames
        ]

        # Change detection validation
        seq_dir = _find_sequence_dir(args.data_dir, seq_id)
        image_paths = get_sorted_frames(seq_dir) if seq_dir is not None else []

        # Build frame_id -> image_path lookup (handles padded sequences)
        image_by_id: dict[str, Path] = {p.stem: p for p in image_paths}

        validated_frames: list[FrameResult] = []
        prev_gray: np.ndarray | None = None

        for fr in frames:
            # Load grayscale image by frame_id
            img_path = image_by_id.get(fr.frame_id)
            curr_gray = (
                cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img_path is not None
                else None
            )

            if img_path is None and fr.detections:
                logger.debug(
                    "No image for frame_id=%s in %s, discarding %d detections",
                    fr.frame_id,
                    seq_id,
                    len(fr.detections),
                )

            if prev_gray is None or curr_gray is None:
                # First frame or missing image: discard detections
                validated_frames.append(
                    FrameResult(
                        frame_id=fr.frame_id,
                        timestamp=fr.timestamp,
                        detections=[],
                    )
                )
                prev_gray = curr_gray
                continue

            # Skip change detection if resolution changed
            if prev_gray.shape != curr_gray.shape:
                logger.warning(
                    "Resolution change in %s at frame %s: %s -> %s",
                    seq_id,
                    fr.frame_id,
                    prev_gray.shape,
                    curr_gray.shape,
                )
                validated_frames.append(
                    FrameResult(
                        frame_id=fr.frame_id,
                        timestamp=fr.timestamp,
                        detections=[],
                    )
                )
                prev_gray = curr_gray
                continue

            # Compute change mask and validate detections
            change_mask = compute_change_mask(
                prev_gray, curr_gray, args.pixel_threshold
            )
            kept = []
            for det in fr.detections:
                ratio = compute_change_ratio_in_bbox(
                    change_mask, det.cx, det.cy, det.w, det.h
                )
                if ratio >= args.min_change_ratio:
                    kept.append(det)

            validated_frames.append(
                FrameResult(
                    frame_id=fr.frame_id,
                    timestamp=fr.timestamp,
                    detections=kept,
                )
            )
            prev_gray = curr_gray

        # Run tracker on change-validated detections
        is_alarm, tracks, confirmed_idx, _frame_traces = tracker.process_sequence(
            validated_frames
        )

        # Look up ground truth
        gt = is_wf_sequence(seq_dir) if seq_dir is not None else False

        total_dets = sum(len(f.detections) for f in validated_frames)
        first_ts = validated_frames[0].timestamp if validated_frames else None
        confirmed_ts = (
            validated_frames[confirmed_idx].timestamp
            if confirmed_idx is not None
            else None
        )

        seq_result = SequenceResult(
            sequence_id=seq_id,
            is_positive_gt=gt,
            is_positive_pred=is_alarm,
            num_frames=len(validated_frames),
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

    tp = sum(1 for r in results if r["is_positive_gt"] and r["is_positive_pred"])
    fp = sum(1 for r in results if not r["is_positive_gt"] and r["is_positive_pred"])
    fn = sum(1 for r in results if r["is_positive_gt"] and not r["is_positive_pred"])
    tn = sum(
        1 for r in results if not r["is_positive_gt"] and not r["is_positive_pred"]
    )
    logger.info("  TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)


if __name__ == "__main__":
    main()

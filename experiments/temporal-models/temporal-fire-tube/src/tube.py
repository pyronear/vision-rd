"""Fire-tube construction: track detections across frames and crop regions.

Implements the adapted fire-tube concept from Park & Ko (2020).  Instead of
building tubes from dense video optical flow, we track YOLO detections across
sparse 30-second-interval frames using greedy IoU matching, then crop and
resize the corresponding image regions into a tube for downstream feature
extraction.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.types import Detection, FireTube, FrameResult, TubeCrop

# ---------------------------------------------------------------------------
# IoU matching (duplicated from tracking-fsm-baseline for isolation)
# ---------------------------------------------------------------------------


def compute_iou(det_a: Detection, det_b: Detection) -> float:
    """Compute Intersection-over-Union between two detections.

    Both detections use normalized center-based coordinates (cx, cy, w, h).
    """
    a_x1 = det_a.cx - det_a.w / 2
    a_y1 = det_a.cy - det_a.h / 2
    a_x2 = det_a.cx + det_a.w / 2
    a_y2 = det_a.cy + det_a.h / 2

    b_x1 = det_b.cx - det_b.w / 2
    b_y1 = det_b.cy - det_b.h / 2
    b_x2 = det_b.cx + det_b.w / 2
    b_y2 = det_b.cy + det_b.h / 2

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = det_a.w * det_a.h
    b_area = det_b.w * det_b.h
    union_area = a_area + b_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_detections(
    prev_dets: list[Detection],
    curr_dets: list[Detection],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching of detections between consecutive frames.

    Returns list of ``(prev_idx, curr_idx, iou)`` tuples.
    """
    if not prev_dets or not curr_dets:
        return []

    pairs = []
    for i, prev in enumerate(prev_dets):
        for j, curr in enumerate(curr_dets):
            iou = compute_iou(prev, curr)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(key=lambda x: -x[0])
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    matches = []

    for iou_val, i, j in pairs:
        if i not in matched_prev and j not in matched_curr:
            matches.append((i, j, iou_val))
            matched_prev.add(i)
            matched_curr.add(j)

    return matches


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------


def crop_detection(
    image_path: Path,
    detection: Detection,
    crop_size: int,
    img_width: int | None = None,
    img_height: int | None = None,
) -> np.ndarray:
    """Crop and resize a detection region from an image.

    Args:
        image_path: Path to the source image file.
        detection: Detection with normalized center-based coordinates.
        crop_size: Target size (square) for the resized crop.
        img_width: Image width override (avoids re-reading for size).
        img_height: Image height override.

    Returns:
        RGB numpy array of shape ``(crop_size, crop_size, 3)``, dtype uint8.
    """
    img = Image.open(image_path).convert("RGB")
    w_px, h_px = img.size
    if img_width is not None:
        w_px = img_width
    if img_height is not None:
        h_px = img_height

    # Convert normalized coords to pixel coords
    x1 = int((detection.cx - detection.w / 2) * w_px)
    y1 = int((detection.cy - detection.h / 2) * h_px)
    x2 = int((detection.cx + detection.w / 2) * w_px)
    y2 = int((detection.cy + detection.h / 2) * h_px)

    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_px, x2)
    y2 = min(h_px, y2)

    # Handle degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    cropped = img.crop((x1, y1, x2, y2))
    resized = cropped.resize((crop_size, crop_size), Image.BILINEAR)
    return np.array(resized, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tube construction
# ---------------------------------------------------------------------------


def _filter_detections(
    frame_results: list[FrameResult],
    confidence_threshold: float,
    max_detection_area: float | None,
) -> list[FrameResult]:
    """Filter detections by confidence and area thresholds."""
    return [
        FrameResult(
            frame_id=fr.frame_id,
            timestamp=fr.timestamp,
            detections=[
                d
                for d in fr.detections
                if d.confidence >= confidence_threshold
                and (max_detection_area is None or d.w * d.h <= max_detection_area)
            ],
        )
        for fr in frame_results
    ]


def build_tubes_for_sequence(
    frame_results: list[FrameResult],
    image_dir: Path,
    sequence_id: str,
    crop_size: int = 64,
    max_tube_length: int = 50,
    confidence_threshold: float = 0.3,
    max_detection_area: float | None = 0.05,
    iou_threshold: float = 0.1,
) -> list[FireTube]:
    """Build fire-tubes from YOLO detections and original images.

    Tracks detections across consecutive frames via greedy IoU matching.
    For each detection chain, crops the corresponding image regions and
    assembles them into a :class:`FireTube`.

    Args:
        frame_results: Per-frame YOLO detections (temporally ordered).
        image_dir: Directory containing the original frame images.
        sequence_id: Identifier for this sequence.
        crop_size: Square size for resized crops.
        max_tube_length: Maximum number of crops per tube.
        confidence_threshold: Minimum detection confidence.
        max_detection_area: Maximum normalized detection area.
        iou_threshold: Minimum IoU for matching across frames.

    Returns:
        List of fire-tubes found in the sequence.
    """
    filtered = _filter_detections(
        frame_results, confidence_threshold, max_detection_area
    )

    # Active tubes: list of (tube_id, last_detection, crops_so_far)
    active: list[tuple[int, Detection, list[TubeCrop]]] = []
    finished: list[FireTube] = []
    next_tube_id = 0

    for frame in filtered:
        curr_dets = frame.detections
        image_path = image_dir / f"{frame.frame_id}.jpg"

        if active and curr_dets:
            prev_dets = [a[1] for a in active]
            matches = match_detections(prev_dets, curr_dets, iou_threshold)
            matched_active_idxs = {m[0] for m in matches}
            matched_det_idxs = {m[1] for m in matches}

            # Update matched tubes
            new_active = []
            for match_prev_idx, match_det_idx, _iou in matches:
                tid, _prev_det, crops = active[match_prev_idx]
                det = curr_dets[match_det_idx]
                crop_img = crop_detection(image_path, det, crop_size)
                crops.append(
                    TubeCrop(
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        image=crop_img,
                        detection=det,
                    )
                )
                # Truncate if too long (keep most recent)
                if len(crops) > max_tube_length:
                    crops = crops[-max_tube_length:]
                new_active.append((tid, det, crops))

            # Finalize unmatched tubes
            for i, (tid, _det, crops) in enumerate(active):
                if i not in matched_active_idxs:
                    finished.append(
                        FireTube(
                            tube_id=tid,
                            sequence_id=sequence_id,
                            crops=crops,
                        )
                    )

            # Create new tubes for unmatched detections
            for j, det in enumerate(curr_dets):
                if j not in matched_det_idxs:
                    crop_img = crop_detection(image_path, det, crop_size)
                    new_active.append(
                        (
                            next_tube_id,
                            det,
                            [
                                TubeCrop(
                                    frame_id=frame.frame_id,
                                    timestamp=frame.timestamp,
                                    image=crop_img,
                                    detection=det,
                                )
                            ],
                        )
                    )
                    next_tube_id += 1

            active = new_active
        else:
            # Finalize all active tubes (no current detections)
            for tid, _det, crops in active:
                finished.append(
                    FireTube(tube_id=tid, sequence_id=sequence_id, crops=crops)
                )
            active = []

            # Create new tubes for all current detections
            for det in curr_dets:
                crop_img = crop_detection(image_path, det, crop_size)
                active.append(
                    (
                        next_tube_id,
                        det,
                        [
                            TubeCrop(
                                frame_id=frame.frame_id,
                                timestamp=frame.timestamp,
                                image=crop_img,
                                detection=det,
                            )
                        ],
                    )
                )
                next_tube_id += 1

    # Finalize remaining active tubes
    for tid, _det, crops in active:
        finished.append(FireTube(tube_id=tid, sequence_id=sequence_id, crops=crops))

    return finished

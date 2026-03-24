from src.types import Detection, FrameResult, Track


def compute_iou(det_a: Detection, det_b: Detection) -> float:
    """Compute IoU between two detections in normalized (cx, cy, w, h) format."""
    # Convert to x1, y1, x2, y2
    a_x1 = det_a.cx - det_a.w / 2
    a_y1 = det_a.cy - det_a.h / 2
    a_x2 = det_a.cx + det_a.w / 2
    a_y2 = det_a.cy + det_a.h / 2

    b_x1 = det_b.cx - det_b.w / 2
    b_y1 = det_b.cy - det_b.h / 2
    b_x2 = det_b.cx + det_b.w / 2
    b_y2 = det_b.cy + det_b.h / 2

    # Intersection
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
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
) -> list[tuple[int, int]]:
    """Greedy matching of detections between consecutive frames.

    Returns list of (prev_idx, curr_idx) matched pairs.
    """
    if not prev_dets or not curr_dets:
        return []

    # Compute all pairwise IoUs
    pairs = []
    for i, prev in enumerate(prev_dets):
        for j, curr in enumerate(curr_dets):
            iou = compute_iou(prev, curr)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    # Greedy: sort by IoU descending, assign greedily
    pairs.sort(key=lambda x: -x[0])
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    matches = []

    for _iou, i, j in pairs:
        if i not in matched_prev and j not in matched_curr:
            matches.append((i, j))
            matched_prev.add(i)
            matched_curr.add(j)

    return matches


class SimpleTracker:
    """Level 0: Simple persistence filter.

    Tracks detections across frames using IoU matching.
    A detection is confirmed if it appears in min_consecutive
    consecutive frames.
    """

    def __init__(self, iou_threshold: float, min_consecutive: int) -> None:
        self.iou_threshold = iou_threshold
        self.min_consecutive = min_consecutive

    def process_sequence(
        self, frames: list[FrameResult]
    ) -> tuple[bool, list[Track], int | None]:
        """Process a sequence of frames.

        Returns:
            is_alarm: True if any track was confirmed.
            tracks: All tracks created during processing.
            confirmed_frame_idx: Frame index where first confirmation occurred.
        """
        active_tracks: list[Track] = []
        all_tracks: list[Track] = []
        next_track_id = 0
        confirmed_frame_idx: int | None = None

        for frame_idx, frame in enumerate(frames):
            curr_dets = frame.detections

            # Match current detections to active tracks
            if active_tracks and curr_dets:
                # Get the latest detection from each active track
                track_dets = [t.hits[-1][1] for t in active_tracks]
                matches = match_detections(track_dets, curr_dets, self.iou_threshold)
                matched_track_idxs = {m[0] for m in matches}
                matched_det_idxs = {m[1] for m in matches}

                # Update matched tracks
                for track_idx, det_idx in matches:
                    track = active_tracks[track_idx]
                    track.hits.append((frame.frame_id, curr_dets[det_idx]))
                    track.consecutive_hits += 1
                    track.consecutive_misses = 0

                # Mark unmatched tracks as missed
                for i, track in enumerate(active_tracks):
                    if i not in matched_track_idxs:
                        track.consecutive_misses += 1
                        track.consecutive_hits = 0

                # Create new tracks for unmatched detections
                for j, det in enumerate(curr_dets):
                    if j not in matched_det_idxs:
                        new_track = Track(
                            track_id=next_track_id,
                            hits=[(frame.frame_id, det)],
                            consecutive_hits=1,
                        )
                        next_track_id += 1
                        active_tracks.append(new_track)
                        all_tracks.append(new_track)
            else:
                # No active tracks or no detections
                # Mark all active tracks as missed
                for track in active_tracks:
                    track.consecutive_misses += 1
                    track.consecutive_hits = 0

                # Create new tracks for all detections
                for det in curr_dets:
                    new_track = Track(
                        track_id=next_track_id,
                        hits=[(frame.frame_id, det)],
                        consecutive_hits=1,
                    )
                    next_track_id += 1
                    active_tracks.append(new_track)
                    all_tracks.append(new_track)

            # Check all active tracks for confirmation
            for track in active_tracks:
                if (
                    not track.confirmed
                    and track.consecutive_hits >= self.min_consecutive
                ):
                    track.confirmed = True
                    track.confirmed_at_frame = frame_idx
                    if confirmed_frame_idx is None:
                        confirmed_frame_idx = frame_idx

            # Remove tracks that have been missing too long (no re-acquisition in L0)
            active_tracks = [t for t in active_tracks if t.consecutive_misses == 0]

        is_alarm = any(t.confirmed for t in all_tracks)
        return is_alarm, all_tracks, confirmed_frame_idx

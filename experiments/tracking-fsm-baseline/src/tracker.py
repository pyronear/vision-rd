"""IoU-based detection tracking with FSM confirmation logic.

Matches detections across consecutive frames using Intersection-over-Union,
then applies a finite-state-machine rule: a track is *confirmed* (alarm raised)
only after it persists for ``min_consecutive`` frames in a row.

Optional post-confirmation filters can reject confirmed tracks based on:
- Mean detection confidence across hits.
- Area change ratio (last / first detection area) to require growth.
"""

import statistics

from src.types import Detection, FrameResult, Track


def pad_sequence(frames: list[FrameResult], min_length: int) -> list[FrameResult]:
    """Pad a short sequence symmetrically by repeating boundary frames.

    If *frames* already has at least *min_length* entries (or is empty),
    it is returned unchanged.  Otherwise the first frame is prepended and
    the last frame is appended in alternation until the list reaches
    *min_length*.

    Returns a new list; the input list is not modified.
    """
    if not frames or len(frames) >= min_length:
        return list(frames)
    result = list(frames)
    prepend = True
    while len(result) < min_length:
        src = frames[0] if prepend else frames[-1]
        result.insert(
            0 if prepend else len(result),
            FrameResult(
                frame_id=src.frame_id,
                timestamp=src.timestamp,
                detections=list(src.detections),
            ),
        )
        prepend = not prepend
    return result


def compute_iou(det_a: Detection, det_b: Detection) -> float:
    """Compute Intersection-over-Union between two detections.

    Both detections must use normalized center-based coordinates
    (cx, cy, w, h) in [0, 1].

    Args:
        det_a: First detection.
        det_b: Second detection.

    Returns:
        IoU value in [0, 1]. Returns 0.0 when the union area is zero.
    """
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

    Computes all pairwise IoUs, sorts by descending IoU, and greedily
    assigns one-to-one matches (each detection used at most once).

    Args:
        prev_dets: Detections from the previous frame.
        curr_dets: Detections from the current frame.
        iou_threshold: Minimum IoU required to consider a pair as a match.

    Returns:
        List of ``(prev_idx, curr_idx)`` index pairs for matched detections.
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

    def __init__(
        self,
        iou_threshold: float,
        min_consecutive: int,
        max_misses: int = 0,
        use_confidence_filter: bool = False,
        min_mean_confidence: float = 0.3,
        use_area_change_filter: bool = False,
        min_area_change: float = 1.1,
    ) -> None:
        """Initialise the tracker.

        Args:
            iou_threshold: Minimum IoU to match a detection to an existing
                track.
            min_consecutive: Number of consecutive frames a track must be
                matched before it is confirmed (alarm raised).
            max_misses: Number of consecutive frames a track can go unmatched
                before it is dropped.  ``0`` means a single miss kills the
                track.
            use_confidence_filter: When ``True``, confirmed tracks whose mean
                detection confidence is below *min_mean_confidence* are
                un-confirmed.
            min_mean_confidence: Minimum mean confidence required to keep a
                confirmed track.  Only used when *use_confidence_filter* is
                ``True``.
            use_area_change_filter: When ``True``, confirmed tracks whose area
                change ratio is below *min_area_change* are un-confirmed.
            min_area_change: Minimum ratio of last-detection area to
                first-detection area.  Only used when *use_area_change_filter*
                is ``True``.
        """
        self.iou_threshold = iou_threshold
        self.min_consecutive = min_consecutive
        self.max_misses = max_misses
        self.use_confidence_filter = use_confidence_filter
        self.min_mean_confidence = min_mean_confidence
        self.use_area_change_filter = use_area_change_filter
        self.min_area_change = min_area_change

    def process_sequence(
        self, frames: list[FrameResult]
    ) -> tuple[bool, list[Track], int | None]:
        """Process a full sequence of frames through the tracker.

        Args:
            frames: Temporally ordered list of per-frame detection results.

        Returns:
            A tuple of ``(is_alarm, tracks, confirmed_frame_idx)`` where:

            - **is_alarm** -- ``True`` if any track was confirmed.
            - **tracks** -- All tracks created during processing.
            - **confirmed_frame_idx** -- Index (into *frames*) where the first
              confirmation occurred, or ``None``.
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

            # Remove tracks that have been missing too long
            active_tracks = [
                t for t in active_tracks if t.consecutive_misses <= self.max_misses
            ]

        # -- Compute features on all tracks (for analysis) --
        for track in all_tracks:
            if track.hits:
                track.mean_confidence = statistics.mean(
                    det.confidence for _, det in track.hits
                )
                first_area = track.hits[0][1].w * track.hits[0][1].h
                last_area = track.hits[-1][1].w * track.hits[-1][1].h
                track.area_change_ratio = (
                    last_area / first_area if first_area > 0 else 0.0
                )

        # -- Post-confirmation filters --
        for track in all_tracks:
            if not track.confirmed:
                continue
            if (
                self.use_confidence_filter
                and track.mean_confidence is not None
                and track.mean_confidence < self.min_mean_confidence
            ):
                track.confirmed = False
                track.confirmed_at_frame = None
                continue
            if (
                self.use_area_change_filter
                and track.area_change_ratio is not None
                and track.area_change_ratio < self.min_area_change
            ):
                track.confirmed = False
                track.confirmed_at_frame = None

        # Recompute alarm after post-filters
        is_alarm = any(t.confirmed for t in all_tracks)
        confirmed_frame_idx = None
        if is_alarm:
            confirmed_frame_idx = min(
                t.confirmed_at_frame
                for t in all_tracks
                if t.confirmed and t.confirmed_at_frame is not None
            )
        return is_alarm, all_tracks, confirmed_frame_idx

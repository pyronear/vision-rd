"""Smoke tube construction via greedy IoU matching across frames.

A *tube* is a chain of YOLO detections across consecutive frames that
correspond to the same spatial smoke region.  Tubes bridge per-frame
detections and the LSTM's need for temporally-ordered features of the
same entity.
"""

from .types import Detection, FrameDetections, Tube, TubeEntry


def compute_iou(det_a: Detection, det_b: Detection) -> float:
    """Compute Intersection-over-Union between two detections.

    Both detections use normalised center-based coordinates
    (cx, cy, w, h) in [0, 1].

    Returns:
        IoU value in [0, 1].  Returns 0.0 when the union area is zero.
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

    Args:
        prev_dets: Detections from the previous frame.
        curr_dets: Detections from the current frame.
        iou_threshold: Minimum IoU required to consider a match.

    Returns:
        List of ``(prev_idx, curr_idx, iou)`` tuples.
    """
    if not prev_dets or not curr_dets:
        return []

    pairs: list[tuple[float, int, int]] = []
    for i, prev in enumerate(prev_dets):
        for j, curr in enumerate(curr_dets):
            iou = compute_iou(prev, curr)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(key=lambda x: -x[0])
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou_val, i, j in pairs:
        if i not in matched_prev and j not in matched_curr:
            matches.append((i, j, iou_val))
            matched_prev.add(i)
            matched_curr.add(j)

    return matches


def build_tubes(
    frame_detections: list[FrameDetections],
    iou_threshold: float = 0.2,
    max_misses: int = 2,
) -> list[Tube]:
    """Build smoke tubes from per-frame detections.

    Tubes are constructed by greedy IoU matching: each detection in
    frame *t* is matched to the closest (by IoU) active tube from
    frame *t-1*.  Unmatched detections start new tubes; unmatched
    tubes accumulate misses and are terminated after *max_misses*
    consecutive gaps.

    Args:
        frame_detections: Temporally ordered per-frame detection lists.
        iou_threshold: Minimum IoU for matching a detection to a tube.
        max_misses: Maximum consecutive gap frames before a tube is
            terminated.

    Returns:
        List of :class:`Tube` objects covering the full sequence.
    """
    if not frame_detections:
        return []

    active_tubes: list[Tube] = []
    finished_tubes: list[Tube] = []
    next_tube_id = 0
    # Track consecutive misses per active tube
    consecutive_misses: list[int] = []

    for frame in frame_detections:
        frame_idx = frame.frame_idx
        curr_dets = frame.detections

        if active_tubes and curr_dets:
            # Get last known detection for each active tube
            last_dets = [_last_detection(tube) for tube in active_tubes]
            matches = match_detections(last_dets, curr_dets, iou_threshold)
            matched_tube_idxs = {m[0] for m in matches}
            matched_det_idxs = {m[1] for m in matches}

            # Extend matched tubes
            for tube_idx, det_idx, _ in matches:
                tube = active_tubes[tube_idx]
                # Fill gap entries for any missed frames
                _fill_gaps(tube, frame_idx)
                tube.entries.append(
                    TubeEntry(frame_idx=frame_idx, detection=curr_dets[det_idx])
                )
                tube.end_frame = frame_idx
                consecutive_misses[tube_idx] = 0

            # Increment misses for unmatched tubes
            for i in range(len(active_tubes)):
                if i not in matched_tube_idxs:
                    consecutive_misses[i] += 1

            # Start new tubes for unmatched detections
            for j, det in enumerate(curr_dets):
                if j not in matched_det_idxs:
                    new_tube = Tube(
                        tube_id=next_tube_id,
                        entries=[TubeEntry(frame_idx=frame_idx, detection=det)],
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                    )
                    next_tube_id += 1
                    active_tubes.append(new_tube)
                    consecutive_misses.append(0)
        else:
            # Increment misses for all active tubes (no detections this frame)
            for i in range(len(active_tubes)):
                consecutive_misses[i] += 1

            # Start new tubes for all detections
            for det in curr_dets:
                new_tube = Tube(
                    tube_id=next_tube_id,
                    entries=[TubeEntry(frame_idx=frame_idx, detection=det)],
                    start_frame=frame_idx,
                    end_frame=frame_idx,
                )
                next_tube_id += 1
                active_tubes.append(new_tube)
                consecutive_misses.append(0)

        # Prune tubes that exceeded max_misses
        surviving_tubes: list[Tube] = []
        surviving_misses: list[int] = []
        for tube, misses in zip(active_tubes, consecutive_misses, strict=True):
            if misses > max_misses:
                finished_tubes.append(tube)
            else:
                surviving_tubes.append(tube)
                surviving_misses.append(misses)
        active_tubes = surviving_tubes
        consecutive_misses = surviving_misses

    # All remaining active tubes are finished
    finished_tubes.extend(active_tubes)

    # Sort by tube_id for deterministic output
    finished_tubes.sort(key=lambda t: t.tube_id)
    return finished_tubes


def _last_detection(tube: Tube) -> Detection:
    """Return the most recent non-gap detection in a tube."""
    for entry in reversed(tube.entries):
        if entry.detection is not None:
            return entry.detection
    raise ValueError(f"Tube {tube.tube_id} has no detections")


def _fill_gaps(tube: Tube, target_frame_idx: int) -> None:
    """Insert gap entries (detection=None) for missed frames."""
    if not tube.entries:
        return
    last_frame = tube.entries[-1].frame_idx
    for gap_idx in range(last_frame + 1, target_frame_idx):
        tube.entries.append(TubeEntry(frame_idx=gap_idx, detection=None))


def interpolate_gaps(tube: Tube) -> Tube:
    """Fill gap entries with a geometrically-interpolated bbox.

    For each entry whose ``detection`` is ``None``:

    * **Interior gap** (observed dets on both sides): linearly interpolate
      ``(cx, cy, w, h)`` between the nearest observed detection before and
      after, using the entry's index as the position parameter.
    * **Boundary gap** (no observation on one side): repeat the nearest
      observed detection on the other side.

    Synthesized detections always carry ``confidence=0.0``. The returned
    tube has ``is_gap=True`` flags on every previously-empty entry.

    Observed entries are left untouched.

    Args:
        tube: Tube whose gap entries (``detection=None``) need filling.

    Returns:
        The same tube object, mutated in place. Returned for chaining.
    """
    observed = [
        (i, e.detection) for i, e in enumerate(tube.entries) if e.detection is not None
    ]
    if not observed:
        return tube

    for i, entry in enumerate(tube.entries):
        if entry.detection is not None:
            continue

        before = next(
            ((j, d) for j, d in reversed(observed) if j < i),
            None,
        )
        after = next(
            ((j, d) for j, d in observed if j > i),
            None,
        )

        if before is not None and after is not None:
            j_b, d_b = before
            j_a, d_a = after
            t = (i - j_b) / (j_a - j_b)
            cx = d_b.cx + t * (d_a.cx - d_b.cx)
            cy = d_b.cy + t * (d_a.cy - d_b.cy)
            w = d_b.w + t * (d_a.w - d_b.w)
            h = d_b.h + t * (d_a.h - d_b.h)
            class_id = d_b.class_id
        elif before is not None:
            d = before[1]
            cx, cy, w, h, class_id = d.cx, d.cy, d.w, d.h, d.class_id
        else:
            assert after is not None
            d = after[1]
            cx, cy, w, h, class_id = d.cx, d.cy, d.w, d.h, d.class_id

        entry.detection = Detection(
            class_id=class_id,
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            confidence=0.0,
        )
        entry.is_gap = True

    return tube


def select_longest_tube(tubes: list[Tube]) -> Tube | None:
    """Pick the single longest tube from a list.

    Length is measured as ``end_frame - start_frame + 1`` (so gaps count
    toward length). Ties are broken by the number of non-gap entries --
    the tube with more real observations wins. If still tied, the first
    in the input order wins.

    Args:
        tubes: Candidate tubes.

    Returns:
        The selected tube, or ``None`` if ``tubes`` is empty.
    """
    if not tubes:
        return None

    def _key(tube: Tube) -> tuple[int, int]:
        length = tube.end_frame - tube.start_frame + 1
        n_observed = sum(1 for e in tube.entries if e.detection is not None)
        return (length, n_observed)

    return max(tubes, key=_key)


def tube_from_record(record: dict) -> Tube:
    """Rebuild a :class:`Tube` from a tube JSON record.

    Inverse of ``_serialize_tube`` in ``scripts/build_tubes.py``. Pure
    function; no I/O.

    Entries with ``bbox=None`` are reconstructed with ``detection=None``
    (pre-interpolation gap shape). Otherwise a :class:`Detection` is
    built from the bbox + confidence; ``confidence=None`` falls back to
    ``0.0``.

    Args:
        record: Parsed tube record. Only the ``tube`` sub-object is
            consulted; other top-level keys are ignored.

    Returns:
        A :class:`Tube` with ``tube_id=0`` (the on-disk dataset is
        single-tube-per-sequence so the id is informational only).
    """
    t = record["tube"]
    entries: list[TubeEntry] = []
    for e in t["entries"]:
        bbox = e["bbox"]
        if bbox is None:
            det: Detection | None = None
        else:
            det = Detection(
                class_id=0,
                cx=bbox[0],
                cy=bbox[1],
                w=bbox[2],
                h=bbox[3],
                confidence=e["confidence"] if e["confidence"] is not None else 0.0,
            )
        entries.append(
            TubeEntry(
                frame_idx=e["frame_idx"],
                detection=det,
                is_gap=e["is_gap"],
            )
        )
    return Tube(
        tube_id=0,
        entries=entries,
        start_frame=t["start_frame"],
        end_frame=t["end_frame"],
    )

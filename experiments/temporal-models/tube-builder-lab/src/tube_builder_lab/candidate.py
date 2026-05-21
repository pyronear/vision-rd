"""The candidate tube builder — EDIT THIS to improve linking.

Pipeline: build tubes with the stock greedy IoU matcher, drop sub-threshold
fragments (same filter the model uses), then merge the fragments that belong to
the same plume back together.

Why a merge pass (grounded in the working-set failure cases): a plume drifts and
grows across 30s-apart frames, so consecutive detections drop below IoU=0.2 and
the greedy matcher splits one plume into several tubes — sometimes overlapping
(parallel tracks), sometimes separated by a detector gap of up to ~9 frames.
Across the working set the fragments to merge are tightly co-located (containment
high, OR centers within ~1 box-size) while genuinely distinct plumes sit far
apart. So two tubes are "the same plume" when they are temporally close AND, at
their closest-in-time observed boxes, the smaller is mostly inside the larger
(IoMin) OR their centers are within ~1 box-size (scale-relative, so tiny boxes
don't chain across many widths and teleport the merged tube).

The merge is then just: group tubes into connected components under that
relation, and rebuild one tube per component.
"""

from __future__ import annotations

from collections.abc import Callable

from bbox_tube_temporal.inference import filter_and_interpolate_tubes
from bbox_tube_temporal.tubes import build_tubes
from bbox_tube_temporal.types import Detection, FrameDetections, Tube, TubeEntry

# Stock builder params (unchanged from the lib defaults).
IOU_THRESHOLD = 0.2
MAX_MISSES = 2

# Inference filter, applied BEFORE merging so the merge sees the same tubes the
# lab displays and can only ever *reduce* the count — it never resurrects
# sub-threshold noise the current builder discards.
MIN_TUBE_LENGTH = 2
MIN_DETECTED_ENTRIES = 2

# "Same plume" thresholds (tuned on the working set; see module docstring).
MERGE_IOMIN = 0.3  # smaller box mostly inside the larger
MERGE_PROX_FACTOR = 1.0  # OR centers within this many box-sizes (scale-relative)
MERGE_MAX_GAP = 10  # frames; bridge re-detections across a detector gap

# (frame_idx, detection) for a tube's real observations, in frame order.
Observed = list[tuple[int, Detection]]


def build_tubes_candidate(frame_detections: list[FrameDetections]) -> list[Tube]:
    tubes = build_tubes(
        frame_detections, iou_threshold=IOU_THRESHOLD, max_misses=MAX_MISSES
    )
    tubes = filter_and_interpolate_tubes(
        tubes,
        min_tube_length=MIN_TUBE_LENGTH,
        min_detected_entries=MIN_DETECTED_ENTRIES,
        interpolate_gaps=False,
    )
    return merge_colocated_tubes(tubes)


def merge_colocated_tubes(tubes: list[Tube]) -> list[Tube]:
    """Merge tubes that are fragments of the same plume.

    Groups tubes into connected components under the :func:`_same_plume`
    relation, rebuilds one tube per component, and re-numbers them by start time.
    """
    observed = [_observed(t) for t in tubes]
    components = _connected_components(
        len(tubes),
        related=lambda i, j: _same_plume(observed[i], observed[j]),
    )
    merged = [_combine(component, observed) for component in components]
    merged = [t for t in merged if t is not None]
    merged.sort(key=lambda t: (t.start_frame, t.end_frame))
    for tube_id, tube in enumerate(merged):
        tube.tube_id = tube_id
    return merged


# --- the "same plume" relation -------------------------------------------------


def _same_plume(a: Observed, b: Observed) -> bool:
    """True if two tubes are fragments of one plume: temporally close, and the
    same box at the frames where they are nearest in time."""
    if not a or not b:
        return False
    if _time_gap(a, b) > MERGE_MAX_GAP:
        return False
    det_a, det_b = _closest_in_time(a, b)
    return _same_box(det_a, det_b)


def _time_gap(a: Observed, b: Observed) -> int:
    """Frames separating the two observed spans (0 when they overlap)."""
    a_start, a_end = a[0][0], a[-1][0]
    b_start, b_end = b[0][0], b[-1][0]
    if b_start > a_end:
        return b_start - a_end
    if a_start > b_end:
        return a_start - b_end
    return 0


def _closest_in_time(a: Observed, b: Observed) -> tuple[Detection, Detection]:
    """The two boxes observed in the nearest frames across the two tubes."""
    best = min(
        ((abs(fa - fb), da, db) for fa, da in a for fb, db in b),
        key=lambda candidate: candidate[0],
    )
    return best[1], best[2]


def _same_box(a: Detection, b: Detection) -> bool:
    """Same physical box across frames: substantial overlap (containment), or
    centers within ~MERGE_PROX_FACTOR box-sizes (scale-relative, no teleporting)."""
    if _iou_min(a, b) >= MERGE_IOMIN:
        return True
    box_size = max(a.w, a.h, b.w, b.h)
    return _center_distance(a, b) <= MERGE_PROX_FACTOR * box_size


def _iou_min(a: Detection, b: Detection) -> float:
    """Intersection over the smaller box's area (containment)."""
    ax1, ay1, ax2, ay2 = a.cx - a.w / 2, a.cy - a.h / 2, a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1, bx2, by2 = b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    smaller = min(a.w * a.h, b.w * b.h)
    return inter / smaller if smaller > 0 else 0.0


def _center_distance(a: Detection, b: Detection) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


# --- grouping and rebuilding ---------------------------------------------------


def _connected_components(
    n: int, related: Callable[[int, int], bool]
) -> list[list[int]]:
    """Partition indices ``range(n)`` into components linked by ``related``."""
    parent = list(range(n))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if related(i, j):
                parent[root(i)] = root(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _combine(members: list[int], observed: list[Observed]) -> Tube | None:
    """Fuse a component's fragments into one tube.

    Keeps the largest box per frame (fuller plume extent when fragments overlap),
    and leaves missed frames as gap placeholders for the pipeline to interpolate.
    """
    largest_by_frame: dict[int, Detection] = {}
    for m in members:
        for frame_idx, det in observed[m]:
            best = largest_by_frame.get(frame_idx)
            if best is None or _area(det) > _area(best):
                largest_by_frame[frame_idx] = det
    if not largest_by_frame:
        return None
    start, end = min(largest_by_frame), max(largest_by_frame)
    entries = [
        TubeEntry(frame_idx=f, detection=largest_by_frame.get(f))
        for f in range(start, end + 1)
    ]
    return Tube(tube_id=0, entries=entries, start_frame=start, end_frame=end)


def _observed(tube: Tube) -> Observed:
    return [(e.frame_idx, e.detection) for e in tube.entries if e.detection is not None]


def _area(det: Detection) -> float:
    return det.w * det.h

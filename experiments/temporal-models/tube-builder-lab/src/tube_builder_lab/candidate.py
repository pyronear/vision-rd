"""The candidate tube builder — EDIT THIS to improve linking.

Approach: build tubes with the stock greedy IoU matcher, then run a post-hoc
agglomerative merge that stitches fragments of the same plume back together.

Why a merge pass (grounded in the working-set failure cases): a plume drifts and
grows across 30s-apart frames, so consecutive detections drop below IoU=0.2 and
the greedy matcher splits one plume into several tubes — sometimes overlapping
(parallel tracks), sometimes separated by a detector gap of up to ~9 frames.
Across the 16 targets the fragments to merge are tightly co-located (containment
``IoMin`` high, OR centers within ~1.5 box-sizes) while genuinely distinct plumes
sit far apart. So we merge two tubes when, at their closest-in-time observed
boxes, the smaller is mostly inside the larger (IoMin) OR their centers are close
relative to the box size (scale-relative, so tiny boxes don't chain across many
widths and teleport the tube), and the temporal gap is small.
"""

from __future__ import annotations

from bbox_tube_temporal.inference import filter_and_interpolate_tubes
from bbox_tube_temporal.tubes import build_tubes
from bbox_tube_temporal.types import Detection, FrameDetections, Tube, TubeEntry

IOU_THRESHOLD = 0.2
MAX_MISSES = 2

# Mirror the model's inference filter, applied BEFORE merging so the merge sees
# the same tubes the lab displays and can only ever *reduce* the count — it never
# resurrects sub-threshold noise the current builder discards.
MIN_TUBE_LENGTH = 2
MIN_DETECTED_ENTRIES = 2

# Post-hoc merge thresholds (see module docstring; tuned on the working set).
MERGE_IOMIN = 0.3  # smaller box mostly inside the larger
MERGE_PROX_FACTOR = 1.0  # OR centers within this many box-sizes (scale-relative)
MERGE_MAX_GAP = 10  # frames; bridge re-detections after a detector gap


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


def _observed(tube: Tube) -> list[tuple[int, Detection]]:
    return [(e.frame_idx, e.detection) for e in tube.entries if e.detection is not None]


def _iomin(a: Detection, b: Detection) -> float:
    """Intersection over the smaller box's area (containment)."""
    ax1, ay1, ax2, ay2 = a.cx - a.w / 2, a.cy - a.h / 2, a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1, bx2, by2 = b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    smaller = min(a.w * a.h, b.w * b.h)
    return inter / smaller if smaller > 0 else 0.0


def _cdist(a: Detection, b: Detection) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _linkable(
    obs_a: list[tuple[int, Detection]], obs_b: list[tuple[int, Detection]]
) -> bool:
    """Same plume? Temporally close AND spatially co-located at their nearest frames."""
    a0, a1 = obs_a[0][0], obs_a[-1][0]
    b0, b1 = obs_b[0][0], obs_b[-1][0]
    if b0 > a1:
        gap = b0 - a1
    elif a0 > b1:
        gap = a0 - b1
    else:
        gap = 0  # overlapping spans
    if gap > MERGE_MAX_GAP:
        return False
    # Spatial test on the closest-in-time observed box from each tube.
    _, da, db = min(
        ((abs(fa - fb), da, db) for fa, da in obs_a for fb, db in obs_b),
        key=lambda x: x[0],
    )
    scale = max(da.w, da.h, db.w, db.h)
    return _iomin(da, db) >= MERGE_IOMIN or _cdist(da, db) <= MERGE_PROX_FACTOR * scale


def merge_colocated_tubes(tubes: list[Tube]) -> list[Tube]:
    """Agglomeratively merge tubes that are fragments of the same plume."""
    obs = [_observed(t) for t in tubes]
    n = len(tubes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if not obs[i]:
            continue
        for j in range(i + 1, n):
            if obs[j] and _linkable(obs[i], obs[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[Tube] = []
    for members in groups.values():
        best_by_frame: dict[int, Detection] = {}
        for m in members:
            for f, d in obs[m]:
                cur = best_by_frame.get(f)
                # When fragments overlap a frame, keep the larger box — it
                # captures more of the plume's extent than a small sub-detection.
                if cur is None or d.w * d.h > cur.w * cur.h:
                    best_by_frame[f] = d
        if not best_by_frame:
            continue
        start, end = min(best_by_frame), max(best_by_frame)
        entries = [
            TubeEntry(frame_idx=f, detection=best_by_frame.get(f))
            for f in range(start, end + 1)
        ]
        merged.append(
            Tube(tube_id=0, entries=entries, start_frame=start, end_frame=end)
        )

    merged.sort(key=lambda t: (t.start_frame, t.end_frame))
    for idx, t in enumerate(merged):
        t.tube_id = idx
    return merged

import pytest
from bbox_tube_temporal.types import Detection, FrameDetections, Tube, TubeEntry

from tube_builder_lab.candidate import build_tubes_candidate, merge_colocated_tubes


def _tube(tid, frame_boxes):
    """frame_boxes: list[(frame_idx, (cx, cy, w, h))]."""
    entries = [
        TubeEntry(
            frame_idx=f,
            detection=Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9),
        )
        for f, (cx, cy, w, h) in frame_boxes
    ]
    return Tube(
        tube_id=tid,
        entries=entries,
        start_frame=frame_boxes[0][0],
        end_frame=frame_boxes[-1][0],
    )


def test_merge_contained_overlapping_fragments():
    # A small box sitting inside a big one over shared frames -> one tube.
    big = _tube(0, [(0, (0.5, 0.5, 0.2, 0.2)), (1, (0.5, 0.5, 0.2, 0.2))])
    small = _tube(1, [(1, (0.5, 0.5, 0.02, 0.02))])
    merged = merge_colocated_tubes([big, small])
    assert len(merged) == 1
    assert (merged[0].start_frame, merged[0].end_frame) == (0, 1)


def test_keep_distinct_far_apart_plumes():
    left = _tube(0, [(0, (0.2, 0.5, 0.05, 0.05)), (1, (0.2, 0.5, 0.05, 0.05))])
    right = _tube(1, [(0, (0.8, 0.5, 0.05, 0.05)), (1, (0.8, 0.5, 0.05, 0.05))])
    merged = merge_colocated_tubes([left, right])
    assert len(merged) == 2


def test_proximity_is_scale_relative():
    # Two tiny (0.02-wide) boxes ~0.75 box-sizes apart -> same object, merge.
    a = _tube(0, [(0, (0.50, 0.5, 0.02, 0.02)), (1, (0.50, 0.5, 0.02, 0.02))])
    near = _tube(1, [(2, (0.515, 0.5, 0.02, 0.02)), (3, (0.515, 0.5, 0.02, 0.02))])
    assert len(merge_colocated_tubes([a, near])) == 1
    # ~1.75 box-sizes apart -> a separate detection, do NOT merge (no teleport).
    far = _tube(1, [(2, (0.535, 0.5, 0.02, 0.02)), (3, (0.535, 0.5, 0.02, 0.02))])
    assert len(merge_colocated_tubes([a, far])) == 2


def test_combine_tiebreak_is_order_invariant():
    # Two co-located tubes share frame 1 with equal-area boxes at different cx;
    # the kept box must be the same regardless of input order (no nondeterminism).
    a = _tube(0, [(0, (0.50, 0.5, 0.02, 0.02)), (1, (0.50, 0.5, 0.02, 0.02))])
    b = _tube(1, [(1, (0.51, 0.5, 0.02, 0.02)), (2, (0.51, 0.5, 0.02, 0.02))])

    def box_at_frame_1(tubes):
        [tube] = tubes
        return next(e.detection for e in tube.entries if e.frame_idx == 1)

    assert box_at_frame_1(merge_colocated_tubes([a, b])).cx == pytest.approx(0.51)
    assert box_at_frame_1(merge_colocated_tubes([b, a])).cx == pytest.approx(0.51)


def test_bridge_gap_redetection_at_same_spot():
    # Same location, re-detected after a 6-frame gap (> max_misses, <= MERGE_MAX_GAP).
    a = _tube(0, [(0, (0.5, 0.5, 0.05, 0.05)), (2, (0.5, 0.5, 0.05, 0.05))])
    b = _tube(1, [(8, (0.5, 0.5, 0.05, 0.05)), (10, (0.5, 0.5, 0.05, 0.05))])
    merged = merge_colocated_tubes([a, b])
    assert len(merged) == 1
    assert (merged[0].start_frame, merged[0].end_frame) == (0, 10)
    # gap frames are placeholders (detection=None) for the pipeline to interpolate
    assert any(e.detection is None for e in merged[0].entries)


def test_do_not_bridge_gap_beyond_window():
    a = _tube(0, [(0, (0.5, 0.5, 0.05, 0.05)), (2, (0.5, 0.5, 0.05, 0.05))])
    b = _tube(1, [(16, (0.5, 0.5, 0.05, 0.05)), (18, (0.5, 0.5, 0.05, 0.05))])
    merged = merge_colocated_tubes([a, b])
    assert len(merged) == 2


def test_transitive_merge_three_fragments():
    a = _tube(0, [(0, (0.5, 0.5, 0.05, 0.05))])
    b = _tube(1, [(3, (0.5, 0.5, 0.05, 0.05))])
    c = _tube(2, [(6, (0.5, 0.5, 0.05, 0.05))])
    merged = merge_colocated_tubes([a, b, c])
    assert len(merged) == 1


def test_build_candidate_merges_grow_split():
    # A plume that stays small for two frames then jumps big for two frames:
    # IoU < 0.2 at the transition so the stock builder splits it into two tubes
    # (each long enough to survive the filter), and the merge pass reunites them.
    def _fd(idx, cx, cy, w, h):
        return FrameDetections(
            frame_idx=idx,
            frame_id=str(idx),
            timestamp=None,
            detections=[Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9)],
        )

    fds = [
        _fd(0, 0.5, 0.5, 0.02, 0.02),
        _fd(1, 0.5, 0.5, 0.02, 0.02),
        _fd(2, 0.5, 0.5, 0.2, 0.2),
        _fd(3, 0.5, 0.5, 0.2, 0.2),
    ]
    tubes = build_tubes_candidate(fds)
    assert len(tubes) == 1

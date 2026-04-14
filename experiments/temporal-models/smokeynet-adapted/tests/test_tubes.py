"""Tests for smokeynet_adapted.tubes."""

from smokeynet_adapted.tubes import (
    build_tubes,
    compute_containment,
    compute_iou,
    match_detections,
    select_longest_tube,
)
from smokeynet_adapted.types import Detection, FrameDetections, Tube, TubeEntry


def _det(cx: float, cy: float, w: float = 0.1, h: float = 0.1, conf: float = 0.8):
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=conf)


def _frame(idx: int, detections: list[Detection]):
    return FrameDetections(
        frame_idx=idx,
        frame_id=f"frame_{idx:03d}",
        timestamp=None,
        detections=detections,
    )


# ── compute_iou ──────────────────────────────────────────────────────────


class TestComputeIou:
    def test_identical_boxes(self):
        det = _det(0.5, 0.5, 0.2, 0.2)
        assert abs(compute_iou(det, det) - 1.0) < 1e-9

    def test_no_overlap(self):
        a = _det(0.1, 0.1, 0.1, 0.1)
        b = _det(0.9, 0.9, 0.1, 0.1)
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap_iou(self):
        a = _det(0.5, 0.5, 0.2, 0.2)
        b = _det(0.55, 0.55, 0.2, 0.2)
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        a = _det(0.5, 0.5, 0.0, 0.0)
        b = _det(0.5, 0.5, 0.1, 0.1)
        assert compute_iou(a, b) == 0.0

    def test_symmetric(self):
        a = _det(0.3, 0.3, 0.2, 0.2)
        b = _det(0.4, 0.4, 0.2, 0.2)
        assert compute_iou(a, b) == compute_iou(b, a)


# ── compute_containment ──────────────────────────────────────────────────


class TestComputeContainment:
    def test_identical_boxes(self):
        det = _det(0.5, 0.5, 0.2, 0.2)
        assert abs(compute_containment(det, det) - 1.0) < 1e-9

    def test_small_inside_large(self):
        """Small GT fully inside large YOLO det -> containment ~1.0."""
        large = _det(0.5, 0.5, 0.2, 0.2)
        small = _det(0.5, 0.5, 0.02, 0.02)
        assert compute_containment(large, small) > 0.99

    def test_small_inside_large_low_iou(self):
        """Same case but IoU is very low due to size mismatch."""
        large = _det(0.5, 0.5, 0.2, 0.2)
        small = _det(0.5, 0.5, 0.02, 0.02)
        assert compute_iou(large, small) < 0.05
        assert compute_containment(large, small) > 0.99

    def test_no_overlap(self):
        a = _det(0.1, 0.1, 0.1, 0.1)
        b = _det(0.9, 0.9, 0.1, 0.1)
        assert compute_containment(a, b) == 0.0

    def test_partial_overlap(self):
        a = _det(0.5, 0.5, 0.2, 0.2)
        b = _det(0.55, 0.55, 0.2, 0.2)
        c = compute_containment(a, b)
        assert 0.0 < c < 1.0

    def test_zero_area(self):
        a = _det(0.5, 0.5, 0.0, 0.0)
        b = _det(0.5, 0.5, 0.1, 0.1)
        assert compute_containment(a, b) == 0.0

    def test_symmetric(self):
        a = _det(0.5, 0.5, 0.2, 0.2)
        b = _det(0.5, 0.5, 0.05, 0.05)
        assert compute_containment(a, b) == compute_containment(b, a)


# ── match_detections ─────────────────────────────────────────────────────


class TestMatchDetections:
    def test_empty_lists(self):
        assert match_detections([], [], 0.1) == []
        assert match_detections([_det(0.5, 0.5)], [], 0.1) == []
        assert match_detections([], [_det(0.5, 0.5)], 0.1) == []

    def test_single_match(self):
        a = _det(0.5, 0.5, 0.2, 0.2)
        b = _det(0.51, 0.51, 0.2, 0.2)
        matches = match_detections([a], [b], 0.1)
        assert len(matches) == 1
        assert matches[0][0] == 0
        assert matches[0][1] == 0

    def test_below_threshold(self):
        a = _det(0.1, 0.1, 0.1, 0.1)
        b = _det(0.9, 0.9, 0.1, 0.1)
        matches = match_detections([a], [b], 0.1)
        assert len(matches) == 0

    def test_greedy_one_to_one(self):
        a1 = _det(0.3, 0.3, 0.2, 0.2)
        a2 = _det(0.7, 0.7, 0.2, 0.2)
        b1 = _det(0.31, 0.31, 0.2, 0.2)
        b2 = _det(0.71, 0.71, 0.2, 0.2)
        matches = match_detections([a1, a2], [b1, b2], 0.1)
        assert len(matches) == 2
        matched_prev = {m[0] for m in matches}
        matched_curr = {m[1] for m in matches}
        assert matched_prev == {0, 1}
        assert matched_curr == {0, 1}


# ── build_tubes ──────────────────────────────────────────────────────────


class TestBuildTubes:
    def test_empty_sequence(self):
        tubes = build_tubes([])
        assert tubes == []

    def test_no_detections(self):
        frames = [_frame(0, []), _frame(1, []), _frame(2, [])]
        tubes = build_tubes(frames)
        assert tubes == []

    def test_single_detection_single_frame(self):
        frames = [_frame(0, [_det(0.5, 0.5)])]
        tubes = build_tubes(frames)
        assert len(tubes) == 1
        assert len(tubes[0].entries) == 1
        assert tubes[0].entries[0].detection is not None

    def test_consistent_detection_forms_one_tube(self):
        """Same bbox across 5 frames should produce a single tube."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [_frame(i, [det]) for i in range(5)]
        tubes = build_tubes(frames, iou_threshold=0.1)
        assert len(tubes) == 1
        assert len(tubes[0].entries) == 5
        assert all(e.detection is not None for e in tubes[0].entries)

    def test_two_separate_detections_form_two_tubes(self):
        """Two non-overlapping detections should form two tubes."""
        d1 = _det(0.2, 0.2, 0.1, 0.1)
        d2 = _det(0.8, 0.8, 0.1, 0.1)
        frames = [_frame(i, [d1, d2]) for i in range(3)]
        tubes = build_tubes(frames, iou_threshold=0.1)
        assert len(tubes) == 2

    def test_gap_creates_none_entries(self):
        """Detection missing in middle frame -> gap entry with None."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [
            _frame(0, [det]),
            _frame(1, []),  # gap
            _frame(2, [det]),
        ]
        tubes = build_tubes(frames, iou_threshold=0.1, max_misses=2)
        assert len(tubes) == 1
        tube = tubes[0]
        # Entries: frame 0 (det), frame 1 (gap), frame 2 (det)
        assert len(tube.entries) == 3
        assert tube.entries[0].detection is not None
        assert tube.entries[1].detection is None
        assert tube.entries[2].detection is not None

    def test_tube_terminated_after_max_misses(self):
        """Tube should be terminated after max_misses consecutive gaps."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [
            _frame(0, [det]),
            _frame(1, []),
            _frame(2, []),
            _frame(3, []),  # 3 consecutive misses
            _frame(4, [det]),  # should start a new tube
        ]
        tubes = build_tubes(frames, iou_threshold=0.1, max_misses=2)
        assert len(tubes) == 2

    def test_max_misses_zero_no_gaps_tolerated(self):
        """With max_misses=0, a single gap terminates the tube."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [
            _frame(0, [det]),
            _frame(1, []),  # gap
            _frame(2, [det]),
        ]
        tubes = build_tubes(frames, iou_threshold=0.1, max_misses=0)
        assert len(tubes) == 2

    def test_tube_start_end_frames(self):
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [_frame(i, [det]) for i in range(5)]
        tubes = build_tubes(frames, iou_threshold=0.1)
        assert tubes[0].start_frame == 0
        assert tubes[0].end_frame == 4

    def test_tube_ids_are_unique(self):
        d1 = _det(0.2, 0.2, 0.1, 0.1)
        d2 = _det(0.8, 0.8, 0.1, 0.1)
        frames = [_frame(i, [d1, d2]) for i in range(3)]
        tubes = build_tubes(frames)
        ids = [t.tube_id for t in tubes]
        assert len(ids) == len(set(ids))

    def test_detection_appears_mid_sequence(self):
        """A detection appearing at frame 3 starts a new tube there."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [
            _frame(0, []),
            _frame(1, []),
            _frame(2, []),
            _frame(3, [det]),
            _frame(4, [det]),
        ]
        tubes = build_tubes(frames, iou_threshold=0.1)
        assert len(tubes) == 1
        assert tubes[0].start_frame == 3
        assert tubes[0].end_frame == 4


# ── select_longest_tube ──────────────────────────────────────────────────


def _make_tube(tube_id: int, length: int, n_gap: int = 0) -> Tube:
    """Build a tube with `length` total entries; last `n_gap` are gaps."""
    n_det = length - n_gap
    entries: list[TubeEntry] = []
    for i in range(n_det):
        entries.append(TubeEntry(frame_idx=i, detection=_det(0.5, 0.5)))
    for i in range(n_gap):
        entries.append(
            TubeEntry(frame_idx=n_det + i, detection=None, is_gap=True)
        )
    return Tube(
        tube_id=tube_id,
        entries=entries,
        start_frame=0,
        end_frame=length - 1,
    )


class TestSelectLongestTube:
    def test_empty_returns_none(self):
        assert select_longest_tube([]) is None

    def test_picks_longest(self):
        a = _make_tube(0, length=3)
        b = _make_tube(1, length=7)
        c = _make_tube(2, length=5)
        result = select_longest_tube([a, b, c])
        assert result is not None
        assert result.tube_id == 1

    def test_tie_break_by_non_gap_count(self):
        # Both span 5 frames; a has 5 dets, b has only 3 (2 gaps).
        a = _make_tube(0, length=5, n_gap=0)
        b = _make_tube(1, length=5, n_gap=2)
        result = select_longest_tube([a, b])
        assert result is not None
        assert result.tube_id == 0

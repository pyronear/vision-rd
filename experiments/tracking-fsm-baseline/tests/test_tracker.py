from datetime import datetime

import pytest

from src.data import pad_sequence
from src.tracker import SimpleTracker, compute_iou, match_detections
from src.types import Detection, FrameResult


def _det(cx: float, cy: float, w: float, h: float, conf: float = 0.9) -> Detection:
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=conf)


def _frame(frame_id: str, detections: list[Detection]) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        detections=detections,
    )


class TestPadSequence:
    def test_no_padding_needed(self):
        frames = [_frame("f1", []), _frame("f2", []), _frame("f3", [])]
        result = pad_sequence(frames, 3)
        assert len(result) == 3

    def test_pads_symmetrically(self):
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        result = pad_sequence(frames, 5)
        assert len(result) == 5
        # Alternates: prepend first, append last, prepend first
        assert result[0].frame_id == "f1"  # prepended
        assert result[1].frame_id == "f1"  # prepended
        assert result[2].frame_id == "f1"  # original
        assert result[3].frame_id == "f2"  # original
        assert result[4].frame_id == "f2"  # appended

    def test_single_frame_padded(self):
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d])]
        result = pad_sequence(frames, 3)
        assert len(result) == 3
        assert all(f.detections == [d] for f in result)

    def test_empty_sequence_unchanged(self):
        result = pad_sequence([], 5)
        assert result == []

    def test_does_not_mutate_input(self):
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d])]
        pad_sequence(frames, 3)
        assert len(frames) == 1

    def test_padded_frames_are_distinct_objects(self):
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        result = pad_sequence(frames, 5)
        # All FrameResult objects should be distinct
        ids = [id(f) for f in result]
        assert len(set(ids)) == len(ids)

    def test_padded_sequence_confirms_with_tracker(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=3)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = pad_sequence([_frame("f1", [d])], 3)
        is_alarm, _tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is True
        assert confirmed_idx == 2


class TestComputeIou:
    def test_identical_boxes(self):
        d = _det(0.5, 0.5, 0.2, 0.2)
        assert compute_iou(d, d) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = _det(0.1, 0.1, 0.1, 0.1)
        b = _det(0.9, 0.9, 0.1, 0.1)
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = _det(0.5, 0.5, 0.2, 0.2)
        b = _det(0.6, 0.5, 0.2, 0.2)
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        a = _det(0.5, 0.5, 0.0, 0.0)
        b = _det(0.5, 0.5, 0.2, 0.2)
        assert compute_iou(a, b) == 0.0


class TestMatchDetections:
    def test_empty_lists(self):
        assert match_detections([], [], 0.3) == []
        assert match_detections([_det(0.5, 0.5, 0.2, 0.2)], [], 0.3) == []
        assert match_detections([], [_det(0.5, 0.5, 0.2, 0.2)], 0.3) == []

    def test_single_match(self):
        d = _det(0.5, 0.5, 0.2, 0.2)
        matches = match_detections([d], [d], 0.3)
        assert matches == [(0, 0)]

    def test_below_threshold(self):
        a = _det(0.1, 0.1, 0.1, 0.1)
        b = _det(0.9, 0.9, 0.1, 0.1)
        assert match_detections([a], [b], 0.3) == []

    def test_greedy_one_to_one(self):
        a1 = _det(0.3, 0.5, 0.2, 0.2)
        a2 = _det(0.7, 0.5, 0.2, 0.2)
        matches = match_detections([a1, a2], [a1, a2], 0.3)
        assert len(matches) == 2
        prev_idxs = {m[0] for m in matches}
        curr_idxs = {m[1] for m in matches}
        assert prev_idxs == {0, 1}
        assert curr_idxs == {0, 1}


class TestSimpleTracker:
    def test_no_detections(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        frames = [_frame("f1", []), _frame("f2", []), _frame("f3", [])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is False
        assert tracks == []
        assert confirmed_idx is None

    def test_single_frame_detection_not_confirmed(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", []), _frame("f3", [])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is False

    def test_consecutive_detections_confirmed(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d]), _frame("f3", [])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is True
        assert confirmed_idx == 1

    def test_min_consecutive_3(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=3)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d]), _frame("f3", [d])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is True
        assert confirmed_idx == 2

    def test_gap_breaks_consecutive(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=3)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [
            _frame("f1", [d]),
            _frame("f2", [d]),
            _frame("f3", []),  # gap
            _frame("f4", [d]),
            _frame("f5", [d]),
        ]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is False

    def test_max_misses_tolerates_gap(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=3, max_misses=1)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [
            _frame("f1", [d]),
            _frame("f2", [d]),
            _frame("f3", []),  # gap — tolerated
            _frame("f4", [d]),
        ]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        # Track survives the gap but consecutive_hits was reset at f3
        # f1: hits=1, f2: hits=2, f3: miss (hits=0), f4: hits=1
        # So with min_consecutive=3, not confirmed
        assert is_alarm is False

    def test_min_consecutive_1_always_confirms(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=1)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is True
        assert confirmed_idx == 0

    def test_track_features_computed(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.8)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _is_alarm, tracks, _idx = tracker.process_sequence(frames)
        assert len(tracks) == 1
        assert tracks[0].mean_confidence == pytest.approx(0.8)
        assert tracks[0].area_change_ratio == pytest.approx(1.0)


class TestConfidenceFilter:
    def test_rejects_low_confidence_track(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_confidence_filter=True,
            min_mean_confidence=0.5,
        )
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.3)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is False
        assert confirmed_idx is None
        assert tracks[0].confirmed is False

    def test_passes_high_confidence_track(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_confidence_filter=True,
            min_mean_confidence=0.5,
        )
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.8)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is True
        assert confirmed_idx == 1

    def test_disabled_flag_does_not_filter(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_confidence_filter=False,
            min_mean_confidence=0.99,
        )
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.3)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        is_alarm, _tracks, _idx = tracker.process_sequence(frames)
        assert is_alarm is True


class TestAreaChangeFilter:
    def test_rejects_static_area_track(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_area_change_filter=True,
            min_area_change=1.1,
        )
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        # area_change_ratio = 1.0 < 1.1 → rejected
        assert is_alarm is False
        assert confirmed_idx is None

    def test_passes_growing_track(self):
        tracker = SimpleTracker(
            iou_threshold=0.1,
            min_consecutive=2,
            use_area_change_filter=True,
            min_area_change=1.1,
        )
        d1 = _det(0.5, 0.5, 0.10, 0.10)
        d2 = _det(0.5, 0.5, 0.15, 0.15)  # area grows 2.25x
        frames = [_frame("f1", [d1]), _frame("f2", [d2])]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        assert is_alarm is True
        assert tracks[0].area_change_ratio == pytest.approx(2.25)

    def test_disabled_flag_does_not_filter(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_area_change_filter=False,
            min_area_change=999.0,
        )
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        is_alarm, _tracks, _idx = tracker.process_sequence(frames)
        assert is_alarm is True


class TestCombinedRules:
    def test_gap_tolerance_plus_confidence_filter(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            max_misses=1,
            use_confidence_filter=True,
            min_mean_confidence=0.5,
        )
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.8)
        frames = [
            _frame("f1", [d]),
            _frame("f2", []),  # gap — tolerated
            _frame("f3", [d]),
            _frame("f4", [d]),
        ]
        is_alarm, tracks, confirmed_idx = tracker.process_sequence(frames)
        # Track survives gap, rebuilds consecutive hits at f3+f4, passes confidence
        assert is_alarm is True

    def test_all_filters_reject(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_confidence_filter=True,
            min_mean_confidence=0.5,
            use_area_change_filter=True,
            min_area_change=1.1,
        )
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.3)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        is_alarm, _tracks, _idx = tracker.process_sequence(frames)
        # Low confidence → rejected by confidence filter
        assert is_alarm is False

from datetime import datetime

import pytest

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

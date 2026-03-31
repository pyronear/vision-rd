from datetime import datetime

import pytest

from mtb_change_detection.data import pad_sequence
from mtb_change_detection.tracker import SimpleTracker, compute_iou, match_detections
from mtb_change_detection.types import Detection, FrameResult


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
        assert result[0].frame_id == "f1"
        assert result[1].frame_id == "f1"
        assert result[2].frame_id == "f1"
        assert result[3].frame_id == "f2"
        assert result[4].frame_id == "f2"

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
        ids = [id(f) for f in result]
        assert len(set(ids)) == len(ids)

    def test_padded_sequence_confirms_with_tracker(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=3)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = pad_sequence([_frame("f1", [d])], 3)
        is_alarm, _tracks, confirmed_idx, _frame_traces = tracker.process_sequence(
            frames
        )
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
        assert len(matches) == 1
        assert matches[0][0] == 0
        assert matches[0][1] == 0
        assert matches[0][2] == pytest.approx(1.0)

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
        assert all(m[2] > 0.3 for m in matches)


class TestSimpleTracker:
    def test_no_detections(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        frames = [_frame("f1", []), _frame("f2", []), _frame("f3", [])]
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
        assert is_alarm is False
        assert tracks == []
        assert confirmed_idx is None

    def test_single_frame_detection_not_confirmed(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", []), _frame("f3", [])]
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
        assert is_alarm is False

    def test_consecutive_detections_confirmed(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d]), _frame("f3", [])]
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
        assert is_alarm is True
        assert confirmed_idx == 1

    def test_min_consecutive_3(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=3)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d]), _frame("f3", [d])]
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
        assert is_alarm is True
        assert confirmed_idx == 2

    def test_gap_breaks_consecutive(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=3)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [
            _frame("f1", [d]),
            _frame("f2", [d]),
            _frame("f3", []),
            _frame("f4", [d]),
            _frame("f5", [d]),
        ]
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
        assert is_alarm is False

    def test_min_consecutive_1_always_confirms(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=1)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d])]
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
        assert is_alarm is True
        assert confirmed_idx == 0

    def test_track_features_computed(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.8)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _is_alarm, tracks, _idx, _ft = tracker.process_sequence(frames)
        assert len(tracks) == 1
        assert tracks[0].mean_confidence == pytest.approx(0.8)
        assert tracks[0].area_change_ratio == pytest.approx(1.0)

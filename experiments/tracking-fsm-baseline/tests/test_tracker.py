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
        # Each match should include an IoU value
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
            _frame("f3", []),  # gap
            _frame("f4", [d]),
            _frame("f5", [d]),
        ]
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
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
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
        # Track survives the gap but consecutive_hits was reset at f3
        # f1: hits=1, f2: hits=2, f3: miss (hits=0), f4: hits=1
        # So with min_consecutive=3, not confirmed
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
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
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
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
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
        is_alarm, _tracks, _idx, _ft = tracker.process_sequence(frames)
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
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
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
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
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
        is_alarm, _tracks, _idx, _ft = tracker.process_sequence(frames)
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
        is_alarm, tracks, confirmed_idx, _ft = tracker.process_sequence(frames)
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
        is_alarm, _tracks, _idx, _ft = tracker.process_sequence(frames)
        # Low confidence → rejected by confidence filter
        assert is_alarm is False


# ---------------------------------------------------------------------------
# Frame trace tests
# ---------------------------------------------------------------------------


class TestFrameTrace:
    def test_trace_has_one_entry_per_frame(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        frames = [_frame("f1", []), _frame("f2", []), _frame("f3", [])]
        _alarm, _tracks, _idx, ft = tracker.process_sequence(frames)
        assert len(ft) == 3
        assert [t.frame_idx for t in ft] == [0, 1, 2]
        assert [t.frame_id for t in ft] == ["f1", "f2", "f3"]

    def test_trace_records_new_tracks(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d])]
        _alarm, _tracks, _idx, ft = tracker.process_sequence(frames)
        assert ft[0].new_track_ids == [0]
        assert ft[0].matches == []

    def test_trace_records_matches_with_iou(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _alarm, _tracks, _idx, ft = tracker.process_sequence(frames)
        assert len(ft[1].matches) == 1
        m = ft[1].matches[0]
        assert m.track_id == 0
        assert m.detection_idx == 0
        assert m.iou == pytest.approx(1.0)

    def test_trace_records_misses(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [])]
        _alarm, _tracks, _idx, ft = tracker.process_sequence(frames)
        assert ft[1].missed_track_ids == [0]

    def test_trace_records_confirmation(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _alarm, _tracks, _idx, ft = tracker.process_sequence(frames)
        assert ft[0].confirmed_track_ids == []
        assert ft[1].confirmed_track_ids == [0]

    def test_trace_records_pruning(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2, max_misses=0)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [])]
        _alarm, _tracks, _idx, ft = tracker.process_sequence(frames)
        assert ft[1].pruned_track_ids == [0]

    def test_trace_num_detections(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d1 = _det(0.3, 0.5, 0.1, 0.1)
        d2 = _det(0.7, 0.5, 0.1, 0.1)
        frames = [_frame("f1", [d1, d2]), _frame("f2", [])]
        _alarm, _tracks, _idx, ft = tracker.process_sequence(frames)
        assert ft[0].num_detections == 2
        assert ft[1].num_detections == 0

    def test_trace_multiple_tracks(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d1 = _det(0.2, 0.5, 0.1, 0.1)
        d2 = _det(0.8, 0.5, 0.1, 0.1)
        frames = [_frame("f1", [d1, d2]), _frame("f2", [d1, d2])]
        _alarm, _tracks, _idx, ft = tracker.process_sequence(frames)
        assert len(ft[0].new_track_ids) == 2
        assert len(ft[1].matches) == 2


# ---------------------------------------------------------------------------
# Post-filter trace tests
# ---------------------------------------------------------------------------


class TestPostFilterTrace:
    def test_confidence_filter_rejection_traced(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_confidence_filter=True,
            min_mean_confidence=0.5,
        )
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.3)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _alarm, tracks, _idx, _ft = tracker.process_sequence(frames)
        assert len(tracks[0].post_filter_results) == 1
        pf = tracks[0].post_filter_results[0]
        assert pf.filter_name == "confidence"
        assert pf.passed is False
        assert pf.actual_value == pytest.approx(0.3)
        assert pf.threshold == 0.5

    def test_confidence_filter_pass_traced(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_confidence_filter=True,
            min_mean_confidence=0.5,
        )
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.8)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _alarm, tracks, _idx, _ft = tracker.process_sequence(frames)
        assert len(tracks[0].post_filter_results) == 1
        pf = tracks[0].post_filter_results[0]
        assert pf.filter_name == "confidence"
        assert pf.passed is True

    def test_area_change_filter_traced(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=2,
            use_area_change_filter=True,
            min_area_change=1.1,
        )
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _alarm, tracks, _idx, _ft = tracker.process_sequence(frames)
        assert len(tracks[0].post_filter_results) == 1
        pf = tracks[0].post_filter_results[0]
        assert pf.filter_name == "area_change"
        assert pf.passed is False
        assert pf.actual_value == pytest.approx(1.0)
        assert pf.threshold == 1.1

    def test_no_filters_yields_empty_results(self):
        tracker = SimpleTracker(iou_threshold=0.3, min_consecutive=2)
        d = _det(0.5, 0.5, 0.2, 0.2)
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _alarm, tracks, _idx, _ft = tracker.process_sequence(frames)
        assert tracks[0].post_filter_results == []

    def test_confidence_rejection_skips_area_filter(self):
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
        _alarm, tracks, _idx, _ft = tracker.process_sequence(frames)
        # Only confidence filter was evaluated (short-circuit)
        assert len(tracks[0].post_filter_results) == 1
        assert tracks[0].post_filter_results[0].filter_name == "confidence"

    def test_both_filters_pass(self):
        tracker = SimpleTracker(
            iou_threshold=0.1,
            min_consecutive=2,
            use_confidence_filter=True,
            min_mean_confidence=0.5,
            use_area_change_filter=True,
            min_area_change=1.1,
        )
        d1 = _det(0.5, 0.5, 0.10, 0.10, conf=0.8)
        d2 = _det(0.5, 0.5, 0.15, 0.15, conf=0.8)
        frames = [_frame("f1", [d1]), _frame("f2", [d2])]
        _alarm, tracks, _idx, _ft = tracker.process_sequence(frames)
        assert len(tracks[0].post_filter_results) == 2
        assert tracks[0].post_filter_results[0].filter_name == "confidence"
        assert tracks[0].post_filter_results[0].passed is True
        assert tracks[0].post_filter_results[1].filter_name == "area_change"
        assert tracks[0].post_filter_results[1].passed is True

    def test_unconfirmed_track_has_no_filter_results(self):
        tracker = SimpleTracker(
            iou_threshold=0.3,
            min_consecutive=3,
            use_confidence_filter=True,
            min_mean_confidence=0.5,
        )
        d = _det(0.5, 0.5, 0.2, 0.2, conf=0.8)
        # Only 2 frames, needs 3 to confirm
        frames = [_frame("f1", [d]), _frame("f2", [d])]
        _alarm, tracks, _idx, _ft = tracker.process_sequence(frames)
        assert tracks[0].confirmed is False
        assert tracks[0].post_filter_results == []

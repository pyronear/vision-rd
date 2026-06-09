"""Unit tests for the pure stable-crop-window helper."""

from types import SimpleNamespace

import pytest

from bbox_tube_temporal.stabilize import tube_union_window, union_window


def _entry(box, is_gap=False):
    """Duck-typed tube entry: ``.detection`` (cx,cy,w,h) or None, plus ``.is_gap``."""
    det = (
        None
        if box is None
        else SimpleNamespace(cx=box[0], cy=box[1], w=box[2], h=box[3])
    )
    return SimpleNamespace(detection=det, is_gap=is_gap)


def test_union_of_two_boxes_is_axis_independent():
    # A: x[0.15,0.25] y[0.45,0.55]; B: x[0.35,0.45] y[0.45,0.55]
    # union: x[0.15,0.45] (w=0.3, cx=0.3); y[0.45,0.55] (h=0.1, cy=0.5)
    boxes = [(0.2, 0.5, 0.1, 0.1), (0.4, 0.5, 0.1, 0.1)]
    assert union_window(boxes) == pytest.approx((0.3, 0.5, 0.3, 0.1))


def test_single_box_returns_itself():
    assert union_window([(0.5, 0.5, 0.2, 0.2)]) == pytest.approx((0.5, 0.5, 0.2, 0.2))


def test_empty_raises():
    with pytest.raises(ValueError):
        union_window([])


def test_tube_union_window_prefers_non_gap_detections():
    # Two real detections + one gap (interpolated) box far to the right. The
    # stabilized window must enclose only the non-gap boxes, ignoring the gap.
    entries = [
        _entry((0.2, 0.5, 0.1, 0.1)),
        _entry((0.9, 0.5, 0.1, 0.1), is_gap=True),
        _entry((0.4, 0.5, 0.1, 0.1)),
    ]
    assert tube_union_window(entries) == pytest.approx((0.3, 0.5, 0.3, 0.1))


def test_tube_union_window_falls_back_to_any_detection_when_all_gaps():
    entries = [
        _entry((0.2, 0.5, 0.1, 0.1), is_gap=True),
        _entry((0.4, 0.5, 0.1, 0.1), is_gap=True),
    ]
    assert tube_union_window(entries) == pytest.approx((0.3, 0.5, 0.3, 0.1))


def test_tube_union_window_none_when_no_detections():
    assert tube_union_window([_entry(None), _entry(None)]) is None

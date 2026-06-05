"""Unit tests for the pure stable-crop-window function."""

from __future__ import annotations

import pytest
from bbox_tube_temporal.types import Detection, Tube, TubeEntry

from tube_builder_lab.stabilize import tube_window


def _det(cx: float, cy: float, w: float, h: float) -> Detection:
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=1.0)


def _tube(*entries: TubeEntry) -> Tube:
    return Tube(
        tube_id=0,
        entries=list(entries),
        start_frame=entries[0].frame_idx,
        end_frame=entries[-1].frame_idx,
    )


def test_union_of_two_boxes():
    # A: x[0.15,0.25] y[0.15,0.25]; B: x[0.35,0.45] y[0.35,0.45]
    # union: x[0.15,0.45] y[0.15,0.45] -> center 0.3,0.3 size 0.3,0.3
    tube = _tube(
        TubeEntry(frame_idx=0, detection=_det(0.2, 0.2, 0.1, 0.1)),
        TubeEntry(frame_idx=1, detection=_det(0.4, 0.4, 0.1, 0.1)),
    )
    assert tube_window(tube) == pytest.approx((0.3, 0.3, 0.3, 0.3))


def test_single_box_tube_returns_that_box():
    tube = _tube(TubeEntry(frame_idx=0, detection=_det(0.5, 0.5, 0.2, 0.2)))
    assert tube_window(tube) == pytest.approx((0.5, 0.5, 0.2, 0.2))


def test_union_is_axis_independent():
    # x extent 0.3, y extent 0.1 -> width and height are computed separately.
    tube = _tube(
        TubeEntry(frame_idx=0, detection=_det(0.2, 0.5, 0.1, 0.1)),
        TubeEntry(frame_idx=1, detection=_det(0.4, 0.5, 0.1, 0.1)),
    )
    assert tube_window(tube) == pytest.approx((0.3, 0.5, 0.3, 0.1))


def test_empty_tube_raises():
    tube = _tube(TubeEntry(frame_idx=0, detection=None, is_gap=True))
    with pytest.raises(ValueError):
        tube_window(tube)


def test_gap_entries_without_detection_are_ignored():
    with_gap = _tube(
        TubeEntry(frame_idx=0, detection=_det(0.2, 0.2, 0.1, 0.1)),
        TubeEntry(frame_idx=1, detection=None, is_gap=True),
        TubeEntry(frame_idx=2, detection=_det(0.4, 0.4, 0.1, 0.1)),
    )
    without_gap = _tube(
        TubeEntry(frame_idx=0, detection=_det(0.2, 0.2, 0.1, 0.1)),
        TubeEntry(frame_idx=1, detection=_det(0.4, 0.4, 0.1, 0.1)),
    )
    assert tube_window(with_gap) == pytest.approx(tube_window(without_gap))

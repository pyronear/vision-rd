from bbox_tube_temporal.types import Detection, Tube, TubeEntry

from tube_builder_lab.viz import (
    bboxes_at_frame,
    norm_bbox_to_pixel,
    tube_color,
    tube_timeline_df,
)


def _entry(idx, cx, cy, conf=0.9, gap=False):
    det = Detection(class_id=0, cx=cx, cy=cy, w=0.2, h=0.4, confidence=conf)
    return TubeEntry(frame_idx=idx, detection=det, is_gap=gap)


def _tube(tid, entries):
    return Tube(
        tube_id=tid,
        entries=entries,
        start_frame=entries[0].frame_idx,
        end_frame=entries[-1].frame_idx,
    )


def test_tube_color_is_stable_and_cyclic():
    assert tube_color(0) == tube_color(10)  # 10-colour palette
    assert tube_color(0) != tube_color(1)


def test_norm_bbox_to_pixel():
    # cx,cy,w,h normalized -> (x0,y0,x1,y1) pixels for a 100x200 image.
    assert norm_bbox_to_pixel((0.5, 0.5, 0.2, 0.4), 100, 200) == (
        40.0,
        60.0,
        60.0,
        140.0,
    )


def test_timeline_df_one_row_per_entry():
    t0 = _tube(0, [_entry(0, 0.5, 0.5), _entry(1, 0.5, 0.5, gap=True)])
    t1 = _tube(1, [_entry(2, 0.2, 0.2)])
    df = tube_timeline_df([t0, t1])
    assert list(df.columns) == ["tube", "frame", "frame_end", "confidence", "is_gap"]
    assert len(df) == 3
    assert set(df["tube"]) == {"T0", "T1"}
    assert df[df["frame"] == 1]["is_gap"].iloc[0]  # the gap entry is flagged


def test_bboxes_at_frame_picks_the_right_entry():
    t0 = _tube(0, [_entry(0, 0.5, 0.5), _entry(1, 0.6, 0.5)])
    t1 = _tube(1, [_entry(1, 0.2, 0.2)])
    got = bboxes_at_frame([t0, t1], 1)
    # (bbox, confidence, tube_id, is_gap) for every tube active at frame 1
    ids = sorted(g[2] for g in got)
    assert ids == [0, 1]
    box_t0 = next(g for g in got if g[2] == 0)[0]
    assert box_t0[0] == 0.6  # cx of t0 at frame 1

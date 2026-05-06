from data_quality_frame_level.audit_app.queue import QueueItem, build_queue
from data_quality_frame_level.audit_app.sequence import assign_temporal_sequences
from data_quality_frame_level.audit_app.types import Prediction
from data_quality_frame_level.dataset import BBox


def _seq_map(stems):
    return assign_temporal_sequences(stems)


def _gt(cx=0.5, cy=0.5, w=0.1, h=0.1):
    return BBox(class_id=0, cx=cx, cy=cy, w=w, h=h)


def _pred(conf, cx=0.5, cy=0.5, w=0.1, h=0.1):
    return Prediction(class_id=0, cx=cx, cy=cy, w=w, h=h, conf=conf)


def test_fp_queue_groups_sequences_by_max_confidence():
    predictions = {
        "seqA_2024-01-01T00-00-00": [_pred(0.7)],
        "seqA_2024-01-01T00-00-30": [_pred(0.9)],
        "seqB_2024-01-01T00-00-00": [_pred(0.6)],
    }
    gt: dict[str, list[BBox]] = {k: [] for k in predictions}
    queue = build_queue(
        predictions=predictions,
        gt=gt,
        sequence_id_by_stem=_seq_map(predictions.keys() | gt.keys()),
        review_status={},
        view="fp",
        conf_thresh=0.05,
        iou_thresh=0.05,
        review_conf_thresh=0.5,
    )
    stems = [item.stem for item in queue]
    assert stems == [
        "seqA_2024-01-01T00-00-00",
        "seqA_2024-01-01T00-00-30",
        "seqB_2024-01-01T00-00-00",
    ]
    assert all(isinstance(item, QueueItem) for item in queue)


def test_fp_queue_filters_by_review_conf():
    stem = "s_2024-01-01T00-00-00"
    predictions = {stem: [_pred(0.4)]}
    gt: dict[str, list[BBox]] = {stem: []}
    out = build_queue(
        predictions=predictions,
        gt=gt,
        sequence_id_by_stem=_seq_map(predictions.keys() | gt.keys()),
        review_status={},
        view="fp",
        conf_thresh=0.05,
        iou_thresh=0.05,
        review_conf_thresh=0.5,
    )
    assert out == []


def test_fn_queue_sorts_by_max_gt_area():
    predictions: dict[str, list[Prediction]] = {
        "seqA_2024-01-01T00-00-00": [],
        "seqB_2024-01-01T00-00-00": [],
    }
    gt = {
        "seqA_2024-01-01T00-00-00": [_gt(w=0.1, h=0.1)],
        "seqB_2024-01-01T00-00-00": [_gt(w=0.3, h=0.3)],
    }
    out = build_queue(
        predictions=predictions,
        gt=gt,
        sequence_id_by_stem=_seq_map(predictions.keys() | gt.keys()),
        review_status={},
        view="fn",
        conf_thresh=0.05,
        iou_thresh=0.05,
        review_conf_thresh=0.0,
    )
    assert [i.stem for i in out] == [
        "seqB_2024-01-01T00-00-00",
        "seqA_2024-01-01T00-00-00",
    ]

import pytest

from data_quality_frame_level.audit_app.matching import (
    EvaluatedFrame,
    evaluate_frame,
    iou,
)
from data_quality_frame_level.audit_app.types import Prediction
from data_quality_frame_level.dataset import BBox


def _gt(cx, cy, w, h):
    return BBox(class_id=0, cx=cx, cy=cy, w=w, h=h)


def _pred(cx, cy, w, h, conf):
    return Prediction(class_id=0, cx=cx, cy=cy, w=w, h=h, conf=conf)


def test_iou_identical():
    b = _gt(0.5, 0.5, 0.2, 0.2)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint():
    a = _gt(0.1, 0.1, 0.1, 0.1)
    b = _gt(0.9, 0.9, 0.1, 0.1)
    assert iou(a, b) == 0.0


def test_evaluate_frame_tp_fp_fn():
    gt = [_gt(0.5, 0.5, 0.2, 0.2), _gt(0.8, 0.8, 0.1, 0.1)]
    preds = [
        _pred(0.5, 0.5, 0.2, 0.2, 0.9),
        _pred(0.1, 0.1, 0.1, 0.1, 0.6),
    ]
    out = evaluate_frame(gt=gt, predictions=preds, iou_thresh=0.5)
    assert isinstance(out, EvaluatedFrame)
    assert out.gt_status == ["tp", "fn"]
    assert out.pred_status == ["tp", "fp"]


def test_evaluate_frame_iou_threshold_filters():
    gt = [_gt(0.5, 0.5, 0.2, 0.2)]
    preds = [_pred(0.6, 0.6, 0.2, 0.2, 0.9)]
    strict = evaluate_frame(gt=gt, predictions=preds, iou_thresh=0.9)
    lenient = evaluate_frame(gt=gt, predictions=preds, iou_thresh=0.05)
    assert strict.gt_status == ["fn"] and strict.pred_status == ["fp"]
    assert lenient.gt_status == ["tp"] and lenient.pred_status == ["tp"]

import pytest

from data_quality_frame_level.audit_app.matching import (
    EvaluatedFrame,
    evaluate_frame,
    iop,
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


def test_iop_pred_fully_inside_gt():
    gt = _gt(0.5, 0.5, 0.4, 0.4)
    pred = _pred(0.5, 0.5, 0.1, 0.1, 0.9)
    assert iop(pred, gt) == pytest.approx(1.0)


def test_iop_pred_half_outside_gt():
    gt = _gt(0.5, 0.5, 0.2, 0.2)
    pred = _pred(0.6, 0.5, 0.2, 0.2, 0.9)
    assert iop(pred, gt) == pytest.approx(0.5)


def test_iop_pred_disjoint_from_gt():
    gt = _gt(0.1, 0.1, 0.1, 0.1)
    pred = _pred(0.9, 0.9, 0.1, 0.1, 0.9)
    assert iop(pred, gt) == 0.0


def test_evaluate_frame_duplicate_preds_all_tp():
    """Many-to-one: 3 overlapping preds on one GT are all TP, no FP."""
    gt = [_gt(0.5, 0.5, 0.2, 0.2)]
    preds = [
        _pred(0.50, 0.50, 0.2, 0.2, 0.9),
        _pred(0.48, 0.50, 0.18, 0.18, 0.8),
        _pred(0.52, 0.50, 0.18, 0.18, 0.7),
    ]
    out = evaluate_frame(gt=gt, predictions=preds, iou_thresh=0.5)
    assert out.gt_status == ["tp"]
    assert out.pred_status == ["tp", "tp", "tp"]


def test_evaluate_frame_containment_absorbs_tight_pred():
    """Tight pred inside generous GT: low IoU, high IoP -> TP only with containment."""
    gt = [_gt(0.5, 0.5, 0.4, 0.4)]
    preds = [_pred(0.5, 0.5, 0.1, 0.1, 0.9)]
    no_contain = evaluate_frame(gt=gt, predictions=preds, iou_thresh=0.5)
    with_contain = evaluate_frame(
        gt=gt,
        predictions=preds,
        iou_thresh=0.5,
        containment_thresh=0.7,
    )
    assert no_contain.gt_status == ["fn"] and no_contain.pred_status == ["fp"]
    assert with_contain.gt_status == ["tp"] and with_contain.pred_status == ["tp"]


def test_evaluate_frame_giant_pred_over_two_gts_is_fp():
    """Huge pred covering two adjacent GTs: low IoU + low IoP -> FP + FN + FN."""
    gt = [_gt(0.30, 0.5, 0.05, 0.05), _gt(0.70, 0.5, 0.05, 0.05)]
    preds = [_pred(0.5, 0.5, 0.9, 0.9, 0.9)]
    out = evaluate_frame(
        gt=gt,
        predictions=preds,
        iou_thresh=0.5,
        containment_thresh=0.7,
    )
    assert out.gt_status == ["fn", "fn"]
    assert out.pred_status == ["fp"]


def test_evaluate_frame_containment_none_is_pure_iou_many_to_one():
    """containment_thresh=None: duplicates TP (many-to-one), tight pred still FP."""
    gt = [_gt(0.5, 0.5, 0.2, 0.2)]
    duplicates = [
        _pred(0.50, 0.50, 0.2, 0.2, 0.9),
        _pred(0.48, 0.50, 0.18, 0.18, 0.8),
    ]
    tight = [_pred(0.5, 0.5, 0.05, 0.05, 0.9)]
    out_dups = evaluate_frame(gt=gt, predictions=duplicates, iou_thresh=0.5)
    out_tight = evaluate_frame(gt=gt, predictions=tight, iou_thresh=0.5)
    assert out_dups.pred_status == ["tp", "tp"]
    assert out_tight.gt_status == ["fn"]
    assert out_tight.pred_status == ["fp"]

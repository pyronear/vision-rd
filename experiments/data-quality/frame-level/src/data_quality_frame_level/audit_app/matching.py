"""Per-frame TP / FP / FN assignment via many-to-one matching.

Each prediction is TP iff at least one GT agrees with it; each GT is TP
iff at least one prediction agrees with it. Agreement is a composite
predicate: ``IoU >= iou_thresh`` OR (when ``containment_thresh`` is
supplied) ``IoP >= containment_thresh``, where ``IoP`` is
intersection-over-prediction-area. Pure-IoU matching is recovered by
passing ``containment_thresh=None`` (the default).
"""

from dataclasses import dataclass

from data_quality_frame_level.audit_app.types import Prediction
from data_quality_frame_level.dataset import BBox


@dataclass(frozen=True)
class EvaluatedFrame:
    gt_status: list[str]
    pred_status: list[str]
    matches: list[tuple[int, int, float]]


def iou(a: BBox | Prediction, b: BBox | Prediction) -> float:
    ax1, ay1 = a.cx - a.w / 2, a.cy - a.h / 2
    ax2, ay2 = a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1 = b.cx - b.w / 2, b.cy - b.h / 2
    bx2, by2 = b.cx + b.w / 2, b.cy + b.h / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def iop(pred: BBox | Prediction, gt: BBox | Prediction) -> float:
    """Intersection over prediction area — 'how much of the pred sits inside the GT?'"""
    px1, py1 = pred.cx - pred.w / 2, pred.cy - pred.h / 2
    px2, py2 = pred.cx + pred.w / 2, pred.cy + pred.h / 2
    gx1, gy1 = gt.cx - gt.w / 2, gt.cy - gt.h / 2
    gx2, gy2 = gt.cx + gt.w / 2, gt.cy + gt.h / 2
    iw = max(0.0, min(px2, gx2) - max(px1, gx1))
    ih = max(0.0, min(py2, gy2) - max(py1, gy1))
    inter = iw * ih
    a = pred.w * pred.h
    return inter / a if a > 0 else 0.0


def evaluate_frame(
    *,
    gt: list[BBox],
    predictions: list[Prediction],
    iou_thresh: float,
    containment_thresh: float | None = None,
) -> EvaluatedFrame:
    pair_iou = [[iou(g, p) for p in predictions] for g in gt]
    pair_iop = [[iop(p, g) for p in predictions] for g in gt]

    def agrees(gi: int, pj: int) -> bool:
        if pair_iou[gi][pj] >= iou_thresh:
            return True
        return containment_thresh is not None and pair_iop[gi][pj] >= containment_thresh

    pred_status = [
        "tp" if any(agrees(gi, pj) for gi in range(len(gt))) else "fp"
        for pj in range(len(predictions))
    ]
    gt_status = [
        "tp" if any(agrees(gi, pj) for pj in range(len(predictions))) else "fn"
        for gi in range(len(gt))
    ]

    matches: list[tuple[int, int, float]] = []
    for gi, st in enumerate(gt_status):
        if st != "tp":
            continue
        best = max(
            (
                (pj, pair_iou[gi][pj])
                for pj in range(len(predictions))
                if agrees(gi, pj)
            ),
            key=lambda x: x[1],
            default=None,
        )
        if best is not None:
            matches.append((gi, best[0], best[1]))
    return EvaluatedFrame(gt_status=gt_status, pred_status=pred_status, matches=matches)

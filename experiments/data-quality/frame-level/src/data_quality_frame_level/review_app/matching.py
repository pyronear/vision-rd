"""Per-frame TP / FP / FN assignment using greedy IoU matching.

Mirrors FiftyOne's ``evaluate_detections`` for our single-class case so
the app is independent of FiftyOne. Predictions and GT are matched
greedily by descending IoU; unmatched predictions become FP, unmatched
GT becomes FN.
"""

from dataclasses import dataclass

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.types import Prediction


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


def evaluate_frame(
    *,
    gt: list[BBox],
    predictions: list[Prediction],
    iou_thresh: float,
) -> EvaluatedFrame:
    candidates = sorted(
        (
            (i, j, iou(g, p))
            for i, g in enumerate(gt)
            for j, p in enumerate(predictions)
        ),
        key=lambda x: x[2],
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gi, pj, score in candidates:
        if score < iou_thresh:
            break
        if gi in matched_gt or pj in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pj)
        matches.append((gi, pj, score))
    gt_status = ["tp" if i in matched_gt else "fn" for i in range(len(gt))]
    pred_status = [
        "tp" if j in matched_pred else "fp" for j in range(len(predictions))
    ]
    return EvaluatedFrame(
        gt_status=gt_status, pred_status=pred_status, matches=matches
    )

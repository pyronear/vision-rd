"""Queue building for the review app.

Filters and sorts the universe of ``(stem, predictions, gt)`` triples
into a list of :class:`QueueItem` for the active view. Sort order
clusters siblings within a sequence (so adjacent items are also
adjacent in time), with sequences ordered by their max severity for
the active view (FP confidence, FN area, or the max of the two for
``all``).
"""

from dataclasses import dataclass

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.matching import evaluate_frame
from data_quality_frame_level.review_app.sequence import parse_stem
from data_quality_frame_level.review_app.types import Prediction


@dataclass(frozen=True)
class QueueItem:
    stem: str
    sequence_id: str
    timestamp: str
    kind: str
    severity: float
    status: str | None


def _frame_severity(
    *,
    gt: list[BBox],
    predictions: list[Prediction],
    conf_thresh: float,
    iou_thresh: float,
    review_conf_thresh: float,
    view: str,
) -> tuple[str | None, float]:
    pred_filt = [p for p in predictions if p.conf >= conf_thresh]
    ev = evaluate_frame(gt=gt, predictions=pred_filt, iou_thresh=iou_thresh)
    fp_conf = max(
        (
            p.conf
            for p, s in zip(pred_filt, ev.pred_status, strict=True)
            if s == "fp" and p.conf >= review_conf_thresh
        ),
        default=0.0,
    )
    fn_area = max(
        (g.w * g.h for g, s in zip(gt, ev.gt_status, strict=True) if s == "fn"),
        default=0.0,
    )
    if view == "fp" and fp_conf > 0.0:
        return "fp", fp_conf
    if view == "fn" and fn_area > 0.0:
        return "fn", fn_area
    if view == "all" and (fp_conf > 0.0 or fn_area > 0.0):
        kind = (
            "mixed"
            if fp_conf > 0.0 and fn_area > 0.0
            else ("fp" if fp_conf > 0.0 else "fn")
        )
        return kind, max(fp_conf, fn_area)
    return None, 0.0


def build_queue(
    *,
    predictions: dict[str, list[Prediction]],
    gt: dict[str, list[BBox]],
    review_status: dict[str, str],
    view: str,
    conf_thresh: float,
    iou_thresh: float,
    review_conf_thresh: float,
) -> list[QueueItem]:
    items: list[QueueItem] = []
    for stem in sorted(predictions.keys() | gt.keys()):
        kind, severity = _frame_severity(
            gt=gt.get(stem, []),
            predictions=predictions.get(stem, []),
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            review_conf_thresh=review_conf_thresh,
            view=view,
        )
        if kind is None:
            continue
        seq_id, ts = parse_stem(stem)
        items.append(
            QueueItem(
                stem=stem,
                sequence_id=seq_id,
                timestamp=ts,
                kind=kind,
                severity=severity,
                status=review_status.get(stem),
            )
        )

    seq_max: dict[str, float] = {}
    for it in items:
        if it.severity > seq_max.get(it.sequence_id, 0.0):
            seq_max[it.sequence_id] = it.severity

    items.sort(key=lambda i: (-seq_max[i.sequence_id], i.sequence_id, i.timestamp))
    return items

"""Build a FiftyOne dataset for frame-level label audit.

For one (model, split) pair: load GT from YOLO label files and YOLO
predictions from predictions.json, attach both as ``fo.Detections`` on a
per-image sample, then call ``dataset.evaluate_detections`` so the
FiftyOne Evaluation panel can filter by TP/FP/FN.

Coordinate conversion goes through :func:`dataset.yolo_to_fiftyone_xywh`
(unit-tested in ``test_dataset.py``); :func:`build_dataset` touches the
live FiftyOne mongo store and is validated end-to-end via ``dvc repro``.
"""

import logging
from collections.abc import Sequence

import fiftyone as fo
from fiftyone import ViewField as F

from data_quality_frame_level.dataset import (
    BBox,
    FrameRef,
    yolo_to_fiftyone_xywh,
)
from data_quality_frame_level.inference import PredBBox
from data_quality_frame_level.review import REVIEW_VOCAB

logger = logging.getLogger(__name__)

SMOKE_LABEL = "smoke"

# Names of the saved views persisted on every dataset. Using identical
# names across train/val/test lets reviewers keep the same view when
# switching datasets via the FiftyOne UI sidebar.
FP_VIEW_NAME = "fp-by-confidence"
FN_VIEW_NAME = "fn-by-area"


def gt_to_detections(bboxes: Sequence[BBox]) -> fo.Detections:
    """Convert ground-truth :class:`BBox` records to ``fo.Detections``."""
    detections = [
        fo.Detection(
            label=SMOKE_LABEL,
            bounding_box=list(yolo_to_fiftyone_xywh(bbox)),
        )
        for bbox in bboxes
    ]
    return fo.Detections(detections=detections)


def preds_to_detections(preds: Sequence[PredBBox]) -> fo.Detections:
    """Convert :class:`PredBBox` records to ``fo.Detections`` with confidence."""
    detections = []
    for pred in preds:
        x = pred.cx - pred.w / 2
        y = pred.cy - pred.h / 2
        detections.append(
            fo.Detection(
                label=SMOKE_LABEL,
                bounding_box=[x, y, pred.w, pred.h],
                confidence=pred.conf,
            )
        )
    return fo.Detections(detections=detections)


def _save_review_views(dataset: fo.Dataset, review_conf_thresh: float) -> None:
    """Persist the FP and FN review views on the dataset.

    YOLO emits every detection above its low per-detection threshold (e.g.
    0.05) so we retain all candidates for interactive review. The FP
    saved view narrows to detections at or above ``review_conf_thresh``
    — reviewers can still slide the confidence filter in the FiftyOne
    sidebar to see lower-confidence FPs without rebuilding the dataset.

    The views are saved with fixed names (:data:`FP_VIEW_NAME`,
    :data:`FN_VIEW_NAME`) so that switching datasets in the FiftyOne UI
    preserves the review workflow.

    - FP view: samples with any FP prediction at conf ≥ review_conf_thresh,
      sorted by the highest qualifying FP confidence descending.
    - FN view: samples with any missed GT bbox, sorted by the largest
      FN-flagged bbox area descending (GTs have no confidence).
    """
    qualifying_fp = F("predictions.detections").filter(
        (F("eval") == "fp") & (F("confidence") >= review_conf_thresh)
    )
    fp_view = dataset.match(qualifying_fp.length() > 0).sort_by(
        qualifying_fp.map(F("confidence")).max(),
        reverse=True,
    )
    fn_view = dataset.match(F("eval_fn") > 0).sort_by(
        F("ground_truth.detections")
        .filter(F("eval") == "fn")
        .map(F("bounding_box")[2] * F("bounding_box")[3])
        .max(),
        reverse=True,
    )
    dataset.save_view(FP_VIEW_NAME, fp_view, overwrite=True)
    dataset.save_view(FN_VIEW_NAME, fn_view, overwrite=True)


def build_dataset(
    dataset_name: str,
    frames: Sequence[FrameRef],
    predictions_by_stem: dict[str, list[PredBBox]],
    iou_thresh: float,
    review_conf_thresh: float,
) -> tuple[fo.Dataset, dict]:
    """Create (or overwrite) a persistent FiftyOne dataset and evaluate it.

    Args:
        dataset_name: Name of the persistent dataset (e.g.
            ``"dq-frame_yolo11s-nimble-narwhal_train"``).
        frames: Frames to include (each contributes one sample).
        predictions_by_stem: Map from :attr:`FrameRef.stem` to the frame's
            predicted bboxes. Stems with no entry are treated as having
            no predictions.
        iou_thresh: Match threshold passed to ``evaluate_detections``.
        review_conf_thresh: Confidence floor used by the saved FP review
            view. Lower values surface more low-confidence flags;
            reviewers can further adjust live in the FiftyOne sidebar
            without rebuilding.

    Returns:
        A tuple ``(dataset, summary)`` where ``summary`` is a dict of
        aggregate counts (TP/FP/FN per sample summed) suitable for
        dumping to ``summary.json``.
    """
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name, persistent=True)

    samples: list[fo.Sample] = []
    for frame in frames:
        preds = predictions_by_stem.get(frame.stem, [])
        samples.append(
            fo.Sample(
                filepath=str(frame.image_path),
                ground_truth=gt_to_detections(frame.gt_bboxes),
                predictions=preds_to_detections(preds),
            )
        )
    dataset.add_samples(samples)

    dataset.evaluate_detections(
        pred_field="predictions",
        gt_field="ground_truth",
        eval_key="eval",
        iou=iou_thresh,
        compute_mAP=False,
    )

    _save_review_views(dataset, review_conf_thresh=review_conf_thresh)

    # Seed the dataset-level tag vocabulary so the FiftyOne tag popover
    # autocompletes the review vocabulary ("label:add-smoke" etc.) as
    # the reviewer types — no need to retype the full tag every time.
    dataset.tags = list(REVIEW_VOCAB)
    dataset.save()

    tp = sum(s.eval_tp for s in dataset)
    fp = sum(s.eval_fp for s in dataset)
    fn = sum(s.eval_fn for s in dataset)
    fp_review_samples = len(dataset.load_saved_view(FP_VIEW_NAME))
    fn_review_samples = len(dataset.load_saved_view(FN_VIEW_NAME))
    logger.info(
        "Saved views on %s: %s (%d samples), %s (%d samples)",
        dataset.name,
        FP_VIEW_NAME,
        fp_review_samples,
        FN_VIEW_NAME,
        fn_review_samples,
    )
    summary = {
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
        "iou_thresh": iou_thresh,
        "review_conf_thresh": review_conf_thresh,
        # Raw bbox-level counts across all detections ≥ the YOLO conf_thresh.
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        # Sample-level review-queue sizes — what reviewers actually walk through.
        "fp_review_samples": fp_review_samples,
        "fn_review_samples": fn_review_samples,
    }
    return dataset, summary

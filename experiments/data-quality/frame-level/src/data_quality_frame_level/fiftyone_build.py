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

from data_quality_frame_level.dataset import (
    BBox,
    FrameRef,
    yolo_to_fiftyone_xywh,
)
from data_quality_frame_level.inference import PredBBox

logger = logging.getLogger(__name__)

SMOKE_LABEL = "smoke"


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


def build_dataset(
    dataset_name: str,
    frames: Sequence[FrameRef],
    predictions_by_stem: dict[str, list[PredBBox]],
    iou_thresh: float,
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

    tp = sum(s.eval_tp for s in dataset)
    fp = sum(s.eval_fp for s in dataset)
    fn = sum(s.eval_fn for s in dataset)
    summary = {
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
        "iou_thresh": iou_thresh,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }
    return dataset, summary

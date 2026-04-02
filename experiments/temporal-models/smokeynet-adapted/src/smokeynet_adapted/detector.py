"""YOLO inference wrapper for smoke detection.

Runs the YOLO model on frames and returns normalised detections.  Image crops
are optionally saved for visualisation but are NOT used for feature extraction
(RoI Align on full-frame feature maps is used instead).
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .types import Detection, FrameDetections


def load_model(model_path: Path) -> YOLO:
    """Load a YOLO model from a checkpoint file.

    Args:
        model_path: Path to the ``.pt`` weights file.

    Returns:
        A loaded :class:`ultralytics.YOLO` instance.
    """
    return YOLO(str(model_path))


def run_yolo_on_frame(
    model: Any,
    image_path: Path,
    conf: float = 0.01,
    iou_nms: float = 0.2,
    img_size: int = 1024,
) -> list[Detection]:
    """Run YOLO inference on a single frame.

    Args:
        model: A loaded YOLO model instance.
        image_path: Path to the frame image.
        conf: Minimum confidence threshold.
        iou_nms: NMS IoU threshold.
        img_size: Input image size for YOLO.

    Returns:
        List of :class:`Detection` objects with normalised coordinates.
    """
    results = model.predict(
        str(image_path),
        conf=conf,
        iou=iou_nms,
        imgsz=img_size,
        verbose=False,
    )
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xywhn = boxes.xywhn.cpu()
        class_ids = boxes.cls.cpu().int()
        confs = boxes.conf.cpu()
        for i in range(len(boxes)):
            detections.append(
                Detection(
                    class_id=int(class_ids[i]),
                    cx=float(xywhn[i, 0]),
                    cy=float(xywhn[i, 1]),
                    w=float(xywhn[i, 2]),
                    h=float(xywhn[i, 3]),
                    confidence=float(confs[i]),
                )
            )
    return detections


def run_yolo_on_sequence(
    model: Any,
    frame_paths: list[Path],
    frame_ids: list[str],
    timestamps: list[datetime | None],
    conf: float = 0.01,
    iou_nms: float = 0.2,
    img_size: int = 1024,
    max_detections_per_frame: int | None = None,
) -> list[FrameDetections]:
    """Run YOLO inference on all frames of a sequence.

    Args:
        model: A loaded YOLO model instance.
        frame_paths: Ordered list of frame image paths.
        frame_ids: Corresponding frame identifiers.
        timestamps: Corresponding timestamps (may be ``None``).
        conf: Minimum confidence threshold.
        iou_nms: NMS IoU threshold.
        img_size: Input image size for YOLO.
        max_detections_per_frame: If set, keep only the top-k highest
            confidence detections per frame.

    Returns:
        List of :class:`FrameDetections`, one per frame.
    """
    results = []
    for idx, (path, fid, ts) in enumerate(
        zip(frame_paths, frame_ids, timestamps, strict=True)
    ):
        dets = run_yolo_on_frame(
            model, path, conf=conf, iou_nms=iou_nms, img_size=img_size
        )
        if (
            max_detections_per_frame is not None
            and len(dets) > max_detections_per_frame
        ):
            dets.sort(key=lambda d: d.confidence, reverse=True)
            dets = dets[:max_detections_per_frame]
        results.append(
            FrameDetections(
                frame_idx=idx,
                frame_id=fid,
                timestamp=ts,
                detections=dets,
            )
        )
    return results

"""Pure-function helpers used by :class:`SmokeynetTemporalModel.predict`.

Each helper corresponds to one stage of the six-stage pipeline described in
``docs/specs/2026-04-15-temporal-model-protocol-design.md``. Kept separate so
``predict()`` is thin and each stage is unit-testable in isolation.
"""

from typing import Any

from pyrocore.types import Frame

from .types import Detection, FrameDetections


def run_yolo_on_frames(
    yolo_model: Any,
    frames: list[Frame],
    *,
    confidence_threshold: float,
    iou_nms: float,
    image_size: int,
) -> list[FrameDetections]:
    """Run YOLO once over all frames in a single batched call.

    Args:
        yolo_model: An ultralytics ``YOLO`` instance (or any object exposing
            ``predict(list_of_paths, ...)`` with the same return shape).
        frames: Temporally ordered Pyronear :class:`Frame` objects.
        confidence_threshold: Minimum detection confidence.
        iou_nms: IoU threshold for YOLO's internal NMS.
        image_size: Inference resolution passed to YOLO.

    Returns:
        One :class:`FrameDetections` per input frame (possibly with zero
        detections), in the same order, with ``frame_idx`` = position.
    """
    if not frames:
        return []

    paths = [str(f.image_path) for f in frames]
    results = yolo_model.predict(
        paths,
        conf=confidence_threshold,
        iou=iou_nms,
        imgsz=image_size,
        verbose=False,
    )

    out: list[FrameDetections] = []
    for idx, (frame, pred) in enumerate(zip(frames, results, strict=True)):
        detections: list[Detection] = []
        boxes = pred.boxes
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn
            confs = boxes.conf
            cls = boxes.cls
            for i in range(len(boxes)):
                row = xywhn[i].tolist()
                detections.append(
                    Detection(
                        class_id=int(cls[i].item()),
                        cx=row[0],
                        cy=row[1],
                        w=row[2],
                        h=row[3],
                        confidence=float(confs[i].item()),
                    )
                )
        out.append(
            FrameDetections(
                frame_idx=idx,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=detections,
            )
        )
    return out

"""YOLO inference wrapper.

Provides functions to load a YOLO model, run it on a sequence of frames,
and serialize / deserialize the per-frame detection results as JSON.
"""

import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from .data import get_sorted_frames, parse_timestamp
from .types import Detection, FrameResult


def load_model(model_path: Path) -> YOLO:
    """Load a YOLO model from a .pt file."""
    return YOLO(str(model_path))


def run_inference_on_frame(
    model: YOLO,
    image_path: Path,
    frame_id: str,
    timestamp: datetime,
    conf: float,
    iou_nms: float,
    img_size: int,
) -> FrameResult:
    """Run YOLO on a single frame image.

    Args:
        model: Loaded YOLO model instance.
        image_path: Path to the frame image file.
        frame_id: Unique identifier for this frame (typically filename stem).
        timestamp: Capture time for this frame.
        conf: Minimum confidence threshold for YOLO predictions.
        iou_nms: IoU threshold used by Non-Maximum Suppression.
        img_size: Input image size (pixels) passed to YOLO.

    Returns:
        A :class:`FrameResult` with normalized center-based detections (xywhn).
    """
    preds = model.predict(
        str(image_path),
        conf=conf,
        iou=iou_nms,
        imgsz=img_size,
        verbose=False,
    )

    detections = []
    for pred in preds:
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            xywhn = boxes.xywhn[i].tolist()
            detections.append(
                Detection(
                    class_id=int(boxes.cls[i].item()),
                    cx=xywhn[0],
                    cy=xywhn[1],
                    w=xywhn[2],
                    h=xywhn[3],
                    confidence=float(boxes.conf[i].item()),
                )
            )

    return FrameResult(
        frame_id=frame_id,
        timestamp=timestamp,
        detections=detections,
    )


def run_inference_on_sequence(
    model: YOLO,
    sequence_dir: Path,
    conf: float,
    iou_nms: float,
    img_size: int,
) -> list[FrameResult]:
    """Run YOLO on all frames in a sequence, return per-frame detections.

    Args:
        model: Loaded YOLO model instance.
        sequence_dir: Path to a sequence directory (must contain ``images/``).
        conf: Minimum confidence threshold for YOLO predictions.
        iou_nms: IoU threshold used by Non-Maximum Suppression.
        img_size: Input image size (pixels) passed to YOLO.

    Returns:
        One :class:`FrameResult` per image, in temporal order. Detections use
        normalized center-based coordinates (xywhn).
    """
    image_paths = get_sorted_frames(sequence_dir)
    return [
        run_inference_on_frame(
            model=model,
            image_path=img_path,
            frame_id=img_path.stem,
            timestamp=parse_timestamp(img_path.name),
            conf=conf,
            iou_nms=iou_nms,
            img_size=img_size,
        )
        for img_path in image_paths
    ]


def save_inference_results(results: list[FrameResult], output_path: Path) -> None:
    """Save per-frame detection results as JSON.

    Args:
        results: List of frame results to serialize.
        output_path: Destination ``.json`` file path (parent dirs are created
            automatically).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for frame in results:
        data.append(
            {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp.isoformat(),
                "detections": [
                    {
                        "class_id": d.class_id,
                        "cx": d.cx,
                        "cy": d.cy,
                        "w": d.w,
                        "h": d.h,
                        "confidence": d.confidence,
                    }
                    for d in frame.detections
                ],
            }
        )
    output_path.write_text(json.dumps(data, indent=2))


def load_inference_results(input_path: Path) -> list[FrameResult]:
    """Load cached inference results from JSON.

    Args:
        input_path: Path to a ``.json`` file written by
            :func:`save_inference_results`.

    Returns:
        List of :class:`FrameResult` objects reconstructed from the JSON.
    """
    data = json.loads(input_path.read_text())
    results = []
    for frame_data in data:
        detections = [
            Detection(
                class_id=d["class_id"],
                cx=d["cx"],
                cy=d["cy"],
                w=d["w"],
                h=d["h"],
                confidence=d["confidence"],
            )
            for d in frame_data["detections"]
        ]
        results.append(
            FrameResult(
                frame_id=frame_data["frame_id"],
                timestamp=datetime.fromisoformat(frame_data["timestamp"]),
                detections=detections,
            )
        )
    return results

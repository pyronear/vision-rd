import json
from pathlib import Path

from ultralytics import YOLO

from src.data import get_sorted_frames, parse_timestamp
from src.types import Detection, FrameResult


def load_model(model_path: Path) -> YOLO:
    """Load a YOLO model from a .pt file."""
    return YOLO(str(model_path))


def run_inference_on_sequence(
    model: YOLO,
    sequence_dir: Path,
    conf: float,
    iou_nms: float,
    img_size: int,
) -> list[FrameResult]:
    """Run YOLO on all frames in a sequence, return per-frame detections."""
    image_paths = get_sorted_frames(sequence_dir)
    results = []

    for img_path in image_paths:
        timestamp = parse_timestamp(img_path.name)
        preds = model.predict(
            str(img_path),
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

        results.append(
            FrameResult(
                frame_id=img_path.stem,
                timestamp=timestamp,
                detections=detections,
            )
        )

    return results


def save_inference_results(results: list[FrameResult], output_path: Path) -> None:
    """Save per-frame detection results as JSON."""
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
    """Load cached inference results from JSON."""
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
                timestamp=parse_timestamp(frame_data["frame_id"]),
                detections=detections,
            )
        )
    return results

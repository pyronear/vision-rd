"""Serialize per-frame YOLO detections to/from the on-disk cache."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from bbox_tube_temporal.types import Detection, FrameDetections


def _det_to_dict(d: Detection) -> dict:
    return {
        "class_id": d.class_id,
        "cx": d.cx,
        "cy": d.cy,
        "w": d.w,
        "h": d.h,
        "confidence": d.confidence,
    }


def _det_from_dict(o: dict) -> Detection:
    return Detection(
        class_id=o["class_id"],
        cx=o["cx"],
        cy=o["cy"],
        w=o["w"],
        h=o["h"],
        confidence=o["confidence"],
    )


def _fd_to_dict(fd: FrameDetections) -> dict:
    return {
        "frame_idx": fd.frame_idx,
        "frame_id": fd.frame_id,
        "timestamp": fd.timestamp.isoformat() if fd.timestamp else None,
        "detections": [_det_to_dict(d) for d in fd.detections],
    }


def _fd_from_dict(o: dict) -> FrameDetections:
    ts = o.get("timestamp")
    return FrameDetections(
        frame_idx=o["frame_idx"],
        frame_id=o["frame_id"],
        timestamp=datetime.fromisoformat(ts) if ts else None,
        detections=[_det_from_dict(d) for d in o["detections"]],
    )


def write_detections(path: Path, frame_detections: list[FrameDetections]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_fd_to_dict(fd) for fd in frame_detections], indent=2))


def read_detections(path: Path) -> list[FrameDetections]:
    return [_fd_from_dict(o) for o in json.loads(path.read_text())]

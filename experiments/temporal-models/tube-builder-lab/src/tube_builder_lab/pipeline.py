"""Turn cached per-frame detections into comparable display tubes.

Both the current and the candidate builders run through the SAME truncation
and the SAME filter, so the only difference between the two sides is the
linking logic itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from bbox_tube_temporal.inference import filter_and_interpolate_tubes
from bbox_tube_temporal.tubes import build_tubes
from bbox_tube_temporal.types import FrameDetections, Tube

Builder = Callable[[list[FrameDetections]], list[Tube]]


@dataclass(frozen=True)
class PipelineConfig:
    max_frames: int
    iou_threshold: float
    max_misses: int
    infer_min_tube_length: int
    min_detected_entries: int
    interpolate_gaps: bool
    confidence_threshold: float
    iou_nms: float
    image_size: int


def extract_pipeline_config(model_config: dict) -> PipelineConfig:
    """Pull the lab-relevant knobs out of a packaged model config.yaml dict."""
    infer = model_config["infer"]
    tubes = model_config["tubes"]
    return PipelineConfig(
        max_frames=int(model_config["classifier"]["max_frames"]),
        iou_threshold=float(tubes["iou_threshold"]),
        max_misses=int(tubes["max_misses"]),
        infer_min_tube_length=int(tubes["infer_min_tube_length"]),
        min_detected_entries=int(tubes["min_detected_entries"]),
        interpolate_gaps=bool(tubes["interpolate_gaps"]),
        confidence_threshold=float(infer["confidence_threshold"]),
        iou_nms=float(infer["iou_nms"]),
        image_size=int(infer["image_size"]),
    )


def write_pipeline_config(path: Path, cfg: PipelineConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.__dict__, default_flow_style=False))


def load_pipeline_config(path: Path) -> PipelineConfig:
    return PipelineConfig(**yaml.safe_load(Path(path).read_text()))


def current_builder(cfg: PipelineConfig) -> Builder:
    """The current lib builder, bound to the model's tube params."""

    def _build(frame_detections: list[FrameDetections]) -> list[Tube]:
        return build_tubes(
            frame_detections,
            iou_threshold=cfg.iou_threshold,
            max_misses=cfg.max_misses,
        )

    return _build


def detections_to_display_tubes(
    frame_detections: list[FrameDetections],
    builder: Builder,
    cfg: PipelineConfig,
    *,
    truncate: bool,
) -> list[Tube]:
    fds = frame_detections[: cfg.max_frames] if truncate else frame_detections
    tubes = builder(fds)
    return filter_and_interpolate_tubes(
        tubes,
        min_tube_length=cfg.infer_min_tube_length,
        min_detected_entries=cfg.min_detected_entries,
        interpolate_gaps=cfg.interpolate_gaps,
    )

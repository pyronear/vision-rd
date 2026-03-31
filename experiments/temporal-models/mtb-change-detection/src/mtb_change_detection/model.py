"""TemporalModel implementation backed by YOLO + MTB change detection + tracker."""

import dataclasses
from datetime import datetime
from typing import Any, Self

import cv2
import numpy as np
from pyrocore import Frame, TemporalModel, TemporalModelOutput

from .change_detector import compute_change_mask, compute_change_ratio_in_bbox
from .data import pad_sequence
from .detector import run_inference_on_frame
from .tracker import SimpleTracker
from .types import FrameResult


class MtbChangeDetectionModel(TemporalModel):
    """Smoke detection model: YOLO + MTB change detection + IoU tracker.

    Implements the pyrocore :class:`TemporalModel` ABC.  The full pipeline is:
    YOLO inference on each frame -> padding -> confidence/area pre-filtering ->
    MTB change validation -> SimpleTracker FSM -> alarm decision.

    The change validation step computes pixel-wise frame differencing between
    consecutive frames and keeps only YOLO detections whose bounding box region
    shows sufficient change (above ``min_change_ratio``).
    """

    def __init__(
        self,
        yolo_model: Any,
        infer_params: dict[str, Any],
        prefilter_params: dict[str, Any],
        change_params: dict[str, Any],
        tracker_params: dict[str, Any],
        min_sequence_length: int = 10,
    ) -> None:
        self._yolo_model = yolo_model
        self._infer_params = infer_params
        self._prefilter_params = prefilter_params
        self._change_params = change_params
        self._tracker_params = tracker_params
        self._min_sequence_length = min_sequence_length

    @classmethod
    def from_params(
        cls,
        yolo_model: Any,
        params: dict[str, Any],
    ) -> Self:
        """Construct from a flat params dict (e.g. loaded from params.yaml)."""
        return cls(
            yolo_model=yolo_model,
            infer_params=params["infer"],
            prefilter_params={
                "confidence_threshold": params["track"]["confidence_threshold"],
                "max_detection_area": params["track"].get("max_detection_area"),
            },
            change_params=params["change"],
            tracker_params={
                k: v
                for k, v in params["track"].items()
                if k not in ("confidence_threshold", "max_detection_area")
            },
            min_sequence_length=params["pad"]["min_sequence_length"],
        )

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        """Run the full YOLO + MTB change detection + tracking pipeline.

        Args:
            frames: Temporally ordered :class:`Frame` objects, as returned by
                :meth:`load_sequence`.

        Returns:
            :class:`TemporalModelOutput` with classification decision and
            metadata in ``details``.
        """
        frame_results = self._run_inference(frames)
        padded = pad_sequence(frame_results, self._min_sequence_length)
        filtered = self._filter_detections(padded)
        change_validated = self._apply_change_validation(frames, filtered)

        tracker = SimpleTracker(**self._tracker_params)
        is_alarm, tracks, confirmed_frame_idx, frame_traces = tracker.process_sequence(
            change_validated
        )

        return TemporalModelOutput(
            is_positive=is_alarm,
            trigger_frame_index=confirmed_frame_idx,
            details={
                "num_tracks": len(tracks),
                "num_confirmed_tracks": sum(1 for t in tracks if t.confirmed),
                "num_detections_total": sum(
                    len(f.detections) for f in change_validated
                ),
                "num_detections_pre_change": sum(len(f.detections) for f in filtered),
                "original_sequence_length": len(frame_results),
                "padded_sequence_length": len(padded),
                "change_params": self._change_params,
                "frame_traces": [dataclasses.asdict(ft) for ft in frame_traces],
                "tracks": [dataclasses.asdict(t) for t in tracks],
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, frames: list[Frame]) -> list[FrameResult]:
        """Run YOLO on each frame and return per-frame detections."""
        conf = self._infer_params["confidence_threshold"]
        iou_nms = self._infer_params["iou_nms"]
        img_size = self._infer_params["image_size"]

        return [
            run_inference_on_frame(
                model=self._yolo_model,
                image_path=frame.image_path,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp or datetime.min,
                conf=conf,
                iou_nms=iou_nms,
                img_size=img_size,
            )
            for frame in frames
        ]

    def _filter_detections(self, frame_results: list[FrameResult]) -> list[FrameResult]:
        """Filter detections by confidence and area thresholds."""
        conf_thresh = self._prefilter_params["confidence_threshold"]
        max_det_area = self._prefilter_params.get("max_detection_area")

        return [
            FrameResult(
                frame_id=fr.frame_id,
                timestamp=fr.timestamp,
                detections=[
                    d
                    for d in fr.detections
                    if d.confidence >= conf_thresh
                    and (max_det_area is None or d.w * d.h <= max_det_area)
                ],
            )
            for fr in frame_results
        ]

    def _apply_change_validation(
        self,
        frames: list[Frame],
        frame_results: list[FrameResult],
    ) -> list[FrameResult]:
        """Filter detections by MTB change detection.

        For each consecutive frame pair, computes a pixel-wise change mask
        and keeps only detections whose bounding box region shows sufficient
        change.

        Detections are discarded (set to empty) when:
        - It is the first frame (no previous frame to compare against).
        - The image cannot be loaded.
        - Consecutive frames have different resolutions.
        - Padded (duplicate) frames compare identical images, yielding
          zero change — this is intentional since synthetic frames should
          not trigger alarms.
        """
        pixel_threshold = self._change_params["pixel_threshold"]
        min_change_ratio = self._change_params["min_change_ratio"]

        # Build frame_id -> image_path lookup (handles padded sequences)
        image_by_id = {f.frame_id: f.image_path for f in frames}

        validated: list[FrameResult] = []
        prev_gray: np.ndarray | None = None

        for fr in frame_results:
            # Load current frame as grayscale by frame_id
            img_path = image_by_id.get(fr.frame_id)
            curr_gray = (
                cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img_path is not None
                else None
            )

            if prev_gray is None or curr_gray is None:
                # First frame or failed load: discard all detections
                validated.append(
                    FrameResult(
                        frame_id=fr.frame_id,
                        timestamp=fr.timestamp,
                        detections=[],
                    )
                )
                prev_gray = curr_gray
                continue

            # Skip if resolution changed between frames
            if prev_gray.shape != curr_gray.shape:
                validated.append(
                    FrameResult(
                        frame_id=fr.frame_id,
                        timestamp=fr.timestamp,
                        detections=[],
                    )
                )
                prev_gray = curr_gray
                continue

            # Compute change mask
            change_mask = compute_change_mask(prev_gray, curr_gray, pixel_threshold)

            # Validate each detection
            kept = []
            for det in fr.detections:
                ratio = compute_change_ratio_in_bbox(
                    change_mask, det.cx, det.cy, det.w, det.h
                )
                if ratio >= min_change_ratio:
                    kept.append(det)

            validated.append(
                FrameResult(
                    frame_id=fr.frame_id,
                    timestamp=fr.timestamp,
                    detections=kept,
                )
            )
            prev_gray = curr_gray

        return validated

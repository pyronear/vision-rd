"""TemporalModel implementation backed by YOLO + FSM tracker."""

from datetime import datetime
from pathlib import Path
from typing import Any, Self

from pyrocore import Frame, TemporalModel, TemporalModelOutput

from src.data import pad_sequence
from src.detector import run_inference_on_frame
from src.package import ModelPackage, load_model_package
from src.tracker import SimpleTracker
from src.types import FrameResult


class FsmTrackingModel(TemporalModel):
    """Smoke detection model: YOLO detector + IoU FSM tracker.

    Implements the pyrocore :class:`TemporalModel` ABC.  The full pipeline is:
    YOLO inference on each frame -> padding -> confidence/area pre-filtering ->
    SimpleTracker FSM -> alarm decision.

    Construct from a packaged archive via :meth:`from_package`, or directly by
    providing the YOLO model instance and config dicts.
    """

    def __init__(
        self,
        yolo_model: Any,
        infer_params: dict[str, Any],
        prefilter_params: dict[str, Any],
        tracker_params: dict[str, Any],
        min_sequence_length: int = 10,
    ) -> None:
        self._yolo_model = yolo_model
        self._infer_params = infer_params
        self._prefilter_params = prefilter_params
        self._tracker_params = tracker_params
        self._min_sequence_length = min_sequence_length

    @classmethod
    def from_package(cls, package_path: Path) -> Self:
        """Load a packaged model archive and return an :class:`FsmTrackingModel`.

        Args:
            package_path: Path to a ``.zip`` archive created by
                :func:`~src.package.build_model_package`.
        """
        pkg: ModelPackage = load_model_package(package_path)
        return cls(
            yolo_model=pkg.model,
            infer_params=pkg.infer_params,
            prefilter_params=pkg.prefilter_params,
            tracker_params=pkg.tracker_params,
            min_sequence_length=pkg.pad_params["min_sequence_length"],
        )

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        """Run the full YOLO + FSM tracking pipeline on a loaded sequence.

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

        tracker = SimpleTracker(**self._tracker_params)
        is_alarm, tracks, confirmed_frame_idx = tracker.process_sequence(filtered)

        return TemporalModelOutput(
            is_positive=is_alarm,
            trigger_frame_index=confirmed_frame_idx,
            details={
                "num_tracks": len(tracks),
                "num_confirmed_tracks": sum(1 for t in tracks if t.confirmed),
                "num_detections_total": sum(len(f.detections) for f in filtered),
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

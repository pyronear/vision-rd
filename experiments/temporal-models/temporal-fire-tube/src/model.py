"""TemporalModel implementation backed by YOLO + fire-tube + Random Forest."""

from datetime import datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
from pyrocore import Frame, TemporalModel, TemporalModelOutput
from sklearn.ensemble import RandomForestClassifier

from src.classifier import predict_tubes
from src.data import pad_sequence
from src.detector import run_inference_on_frame
from src.features import extract_tabular_features
from src.package import ModelPackage, load_model_package
from src.tube import build_tubes_for_sequence
from src.types import FrameResult


class FireTubeModel(TemporalModel):
    """Smoke detection model: YOLO detector + fire-tube + Random Forest.

    Implements the pyrocore :class:`TemporalModel` ABC.  The full pipeline is:
    YOLO inference on each frame -> padding -> tube construction (IoU tracking
    + crop extraction) -> tabular feature extraction -> RF classification ->
    alarm decision.

    Construct from a packaged archive via :meth:`from_package`, or directly by
    providing the YOLO model instance, classifier, and config dicts.
    """

    def __init__(
        self,
        yolo_model: Any,
        classifier: RandomForestClassifier,
        infer_params: dict[str, Any],
        tube_params: dict[str, Any],
        min_sequence_length: int = 10,
    ) -> None:
        self._yolo_model = yolo_model
        self._classifier = classifier
        self._infer_params = infer_params
        self._tube_params = tube_params
        self._min_sequence_length = min_sequence_length

    @classmethod
    def from_package(cls, package_path: Path) -> Self:
        """Load a packaged model archive and return a :class:`FireTubeModel`."""
        pkg: ModelPackage = load_model_package(package_path)
        return cls(
            yolo_model=pkg.model,
            classifier=pkg.classifier,
            infer_params=pkg.infer_params,
            tube_params=pkg.tube_params,
            min_sequence_length=pkg.pad_params["min_sequence_length"],
        )

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        """Run the full YOLO + fire-tube + RF pipeline on a loaded sequence.

        Args:
            frames: Temporally ordered :class:`Frame` objects.

        Returns:
            :class:`TemporalModelOutput` with classification decision and
            metadata in ``details``.
        """
        frame_results = self._run_inference(frames)
        padded = pad_sequence(frame_results, self._min_sequence_length)

        # Build fire-tubes from detections + original images
        # Use the directory of the first frame as the image directory
        image_dir = frames[0].image_path.parent if frames else Path(".")
        tubes = build_tubes_for_sequence(
            frame_results=padded,
            image_dir=image_dir,
            sequence_id="predict",
            crop_size=self._tube_params.get("crop_size", 64),
            max_tube_length=self._tube_params.get("max_tube_length", 50),
            confidence_threshold=self._tube_params.get("confidence_threshold", 0.3),
            max_detection_area=self._tube_params.get("max_detection_area", 0.05),
            iou_threshold=self._tube_params.get("iou_threshold", 0.1),
        )

        # Extract features and classify each tube
        is_alarm = False
        confirmed_frame_idx = None
        num_positive_tubes = 0

        if tubes:
            features = np.array([extract_tabular_features(t) for t in tubes])
            predictions, confidences = predict_tubes(features, self._classifier)

            for pred, tube in zip(predictions, tubes, strict=True):
                if pred:
                    num_positive_tubes += 1
                    # Trigger frame = last frame of the earliest positive tube
                    last_crop = tube.crops[-1]
                    # Find the frame index in the padded sequence
                    for fi, fr in enumerate(padded):
                        if fr.frame_id == last_crop.frame_id:
                            if confirmed_frame_idx is None or fi < confirmed_frame_idx:
                                confirmed_frame_idx = fi
                            break
                    is_alarm = True

        return TemporalModelOutput(
            is_positive=is_alarm,
            trigger_frame_index=confirmed_frame_idx,
            details={
                "num_tubes": len(tubes),
                "num_positive_tubes": num_positive_tubes,
                "num_detections_total": sum(len(f.detections) for f in padded),
                "original_sequence_length": len(frame_results),
                "padded_sequence_length": len(padded),
                "tube_lengths": [len(t.crops) for t in tubes],
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

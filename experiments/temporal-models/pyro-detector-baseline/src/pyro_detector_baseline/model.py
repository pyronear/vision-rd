"""TemporalModel implementation backed by pyro-predictor."""

from pathlib import Path

from PIL import Image
from pyrocore import Frame, TemporalModel, TemporalModelOutput

from .predictor_wrapper import create_predictor


class PyroDetectorModel(TemporalModel):
    """Production smoke detection baseline using pyro-predictor.

    Wraps the pyro-engine Predictor (YOLO ONNX + sliding-window temporal
    smoothing) as a :class:`TemporalModel` subclass for comparison with
    other temporal models.

    The Predictor maintains per-camera sliding-window state.  Each call
    to :meth:`predict` uses a unique camera ID so sequences are isolated.
    The alarm threshold is the Predictor's own ``conf_thresh``, matching
    production behavior.
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_thresh: float = 0.35,
        model_conf_thresh: float = 0.05,
        nb_consecutive_frames: int = 7,
        max_bbox_size: float = 0.4,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        self._predictor = create_predictor(
            model_path=model_path,
            conf_thresh=conf_thresh,
            model_conf_thresh=model_conf_thresh,
            nb_consecutive_frames=nb_consecutive_frames,
            max_bbox_size=max_bbox_size,
            frame_size=frame_size,
        )
        self._sequence_counter = 0

    @classmethod
    def from_model_dir(cls, model_dir: Path, **kwargs) -> "PyroDetectorModel":
        """Load from a directory containing the extracted ONNX model.

        Args:
            model_dir: Directory containing the ONNX model files.
            **kwargs: Additional arguments forwarded to the constructor.
        """
        onnx_files = [
            f for f in model_dir.glob("**/*.onnx") if not f.name.startswith("._")
        ]
        model_path = str(onnx_files[0]) if onnx_files else None
        return cls(model_path=model_path, **kwargs)

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        """Run pyro-predictor on a loaded sequence.

        Args:
            frames: Temporally ordered :class:`Frame` objects.

        Returns:
            :class:`TemporalModelOutput` with classification decision and
            per-frame confidences in ``details``.
        """
        cam_id = f"seq_{self._sequence_counter}"
        self._sequence_counter += 1

        threshold = self._predictor.conf_thresh
        confidences: list[float] = []
        trigger_frame_index: int | None = None

        for i, frame in enumerate(frames):
            pil_img = Image.open(frame.image_path)
            confidence = self._predictor.predict(pil_img, cam_id=cam_id)
            confidences.append(float(confidence))

            if confidence > threshold and trigger_frame_index is None:
                trigger_frame_index = i

        is_positive = trigger_frame_index is not None

        return TemporalModelOutput(
            is_positive=is_positive,
            trigger_frame_index=trigger_frame_index,
            details={
                "per_frame_confidences": confidences,
                "conf_thresh": threshold,
                "num_frames": len(frames),
                "num_detections_total": sum(1 for c in confidences if c > 0),
                "max_confidence": max(confidences) if confidences else 0.0,
            },
        )

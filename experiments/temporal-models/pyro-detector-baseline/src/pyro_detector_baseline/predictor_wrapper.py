"""Wrapper around pyro-predictor's Predictor for sequence-level evaluation.

Provides factory and helper functions to run the production Predictor on
a sequence of frames and collect per-frame confidences.
"""

from pathlib import Path

import pyro_predictor
from PIL import Image


def create_predictor(
    model_path: str | None = None,
    conf_thresh: float = 0.15,
    model_conf_thresh: float = 0.05,
    nb_consecutive_frames: int = 8,
    max_bbox_size: float = 0.4,
    frame_size: tuple[int, int] | None = None,
):
    """Create a pyro-predictor Predictor instance.

    Args:
        model_path: Path to an ONNX model file, or ``None`` to use the
            default model downloaded from HuggingFace Hub.
        conf_thresh: Confidence threshold for active alerts.
        model_conf_thresh: Per-frame YOLO confidence threshold.
        nb_consecutive_frames: Temporal sliding window size.
        max_bbox_size: Maximum detection width as image fraction.
        frame_size: Optional ``(width, height)`` to resize frames.

    Returns:
        A ``pyro_predictor.Predictor`` instance.
    """
    return pyro_predictor.Predictor(
        model_path=model_path,
        conf_thresh=conf_thresh,
        model_conf_thresh=model_conf_thresh,
        nb_consecutive_frames=nb_consecutive_frames,
        max_bbox_size=max_bbox_size,
        frame_size=frame_size,
        verbose=False,
    )


def predict_sequence(
    predictor,
    frame_paths: list[Path],
    cam_id: str,
) -> tuple[bool, int | None, list[float]]:
    """Run the predictor on all frames in a sequence.

    Uses a unique *cam_id* per sequence so the Predictor's internal
    sliding-window state is isolated.  The alarm threshold is taken from
    the predictor's own ``conf_thresh`` to match production behavior.

    Args:
        predictor: A ``pyro_predictor.Predictor`` instance.
        frame_paths: Temporally ordered image paths.
        cam_id: Unique identifier for this sequence (used as camera ID).

    Returns:
        A tuple of ``(is_alarm, trigger_frame_index, per_frame_confidences)``.
    """
    threshold = predictor.conf_thresh
    confidences: list[float] = []
    trigger_frame_index: int | None = None

    for i, frame_path in enumerate(frame_paths):
        pil_img = Image.open(frame_path)
        confidence = predictor.predict(pil_img, cam_id=cam_id)
        confidences.append(float(confidence))

        if confidence > threshold and trigger_frame_index is None:
            trigger_frame_index = i

    is_alarm = trigger_frame_index is not None
    return is_alarm, trigger_frame_index, confidences

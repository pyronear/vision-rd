"""Wrapper around pyro-predictor's Predictor for sequence-level evaluation.

Provides factory and helper functions to run the production Predictor on
a sequence of frames and collect per-frame confidences.

The ``create_predictor`` factory creates a full Predictor (with YOLO model)
for live inference.  The ``create_replay_predictor``, ``load_detections``,
and ``replay_sequence`` helpers support offline replay of cached YOLO
detections through the Predictor's temporal logic without loading the model.
"""

import json
from pathlib import Path

import numpy as np
import pyro_predictor
from PIL import Image
from pyro_predictor.predictor import Predictor

_DUMMY_FRAME = Image.new("RGB", (1, 1))


def create_predictor(
    model_path: str | None = None,
    conf_thresh: float = 0.35,
    model_conf_thresh: float = 0.05,
    nb_consecutive_frames: int = 7,
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


def create_replay_predictor(
    conf_thresh: float,
    nb_consecutive_frames: int,
) -> Predictor:
    """Create a lightweight Predictor for temporal replay without loading YOLO.

    Uses ``object.__new__`` to skip ``__init__`` (and ONNX model loading),
    setting only the attributes that ``_update_states`` and ``_new_state``
    require: ``conf_thresh``, ``nb_consecutive_frames``, and ``_states``.
    """
    replay = object.__new__(Predictor)
    replay.conf_thresh = conf_thresh
    replay.nb_consecutive_frames = nb_consecutive_frames
    replay._states = {}
    return replay


def load_detections(json_path: Path) -> list[tuple[str, np.ndarray]]:
    """Load cached per-frame detections from an infer JSON.

    Args:
        json_path: Path to a sequence JSON produced by the infer stage.

    Returns:
        List of ``(filename, detections)`` tuples where *detections* is
        a ``(N, 5)`` numpy array of ``[x1, y1, x2, y2, conf]``.
    """
    data = json.loads(json_path.read_text())
    return [
        (d["filename"], np.array(d["detections"], dtype=np.float64).reshape(-1, 5))
        for d in data
    ]


def replay_sequence(
    predictor: Predictor,
    frame_detections: list[tuple[str, np.ndarray]],
    cam_key: str,
) -> tuple[int | None, list[float]]:
    """Replay cached detections through the Predictor's temporal logic.

    Args:
        predictor: A replay Predictor created via ``create_replay_predictor``.
        frame_detections: Per-frame detections from ``load_detections``.
        cam_key: Unique identifier for this sequence (used as camera ID).

    Returns:
        A tuple of ``(trigger_frame_index, per_frame_confidences)``.
        ``trigger_frame_index`` is ``None`` if no alarm was triggered.
    """
    predictor._states[cam_key] = predictor._new_state()

    confidences: list[float] = []
    trigger_idx: int | None = None
    for i, (_filename, preds) in enumerate(frame_detections):
        conf = predictor._update_states(_DUMMY_FRAME, preds, cam_key)
        confidences.append(float(conf))
        if conf > predictor.conf_thresh and trigger_idx is None:
            trigger_idx = i

    del predictor._states[cam_key]
    return trigger_idx, confidences


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

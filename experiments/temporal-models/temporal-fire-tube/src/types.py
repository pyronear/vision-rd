"""Core data types for the temporal-fire-tube pipeline.

Defines the dataclasses used across inference, tube construction, feature
extraction, classification, and evaluation stages.  All bounding-box
coordinates use a normalized center-based format (values in [0, 1]).
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Detection:
    """A single YOLO detection in a frame.

    Bounding-box coordinates are normalized and center-based: ``cx`` and ``cy``
    give the box center, ``w`` and ``h`` its width and height, all relative to
    image dimensions (values in [0, 1]).
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float


@dataclass
class FrameResult:
    """All detections produced by YOLO for a single camera frame."""

    frame_id: str
    timestamp: datetime
    detections: list[Detection]


@dataclass
class SequenceResult:
    """Per-sequence evaluation record joining prediction and ground truth."""

    sequence_id: str
    is_positive_gt: bool
    is_positive_pred: bool
    num_frames: int
    num_detections_total: int
    num_tubes: int
    confirmed_frame_index: int | None = None
    confirmed_timestamp: datetime | None = None
    first_timestamp: datetime | None = None


# --- Fire-tube types ---


@dataclass
class TubeCrop:
    """A single crop from one frame within a fire-tube."""

    frame_id: str
    timestamp: datetime
    image: np.ndarray  # shape (crop_size, crop_size, 3), uint8
    detection: Detection


@dataclass
class FireTube:
    """A tracked detection across consecutive frames, with cropped regions.

    Each tube represents one candidate object tracked via IoU across frames.
    The ``crops`` list contains the cropped and resized image regions for each
    frame where the detection was matched.
    """

    tube_id: int
    sequence_id: str
    crops: list[TubeCrop] = field(default_factory=list)
    label: bool | None = None  # ground-truth label (set during training)

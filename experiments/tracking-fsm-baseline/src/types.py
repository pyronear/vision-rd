"""Core data types for the tracking-fsm-baseline pipeline.

Defines the dataclasses used across inference, tracking, and evaluation stages.
All bounding-box coordinates use a normalized center-based format (values in [0, 1]).
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Detection:
    """A single YOLO detection in a frame.

    Bounding-box coordinates are normalized and center-based: ``cx`` and ``cy``
    give the box center, ``w`` and ``h`` its width and height, all relative to
    image dimensions (values in [0, 1]).

    Attributes:
        class_id: Integer class label predicted by YOLO.
        cx: Normalized center-x of the bounding box.
        cy: Normalized center-y of the bounding box.
        w: Normalized width of the bounding box.
        h: Normalized height of the bounding box.
        confidence: Detection confidence score in [0, 1].
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float


@dataclass
class FrameResult:
    """All detections produced by YOLO for a single camera frame.

    Attributes:
        frame_id: Unique identifier derived from the image filename (stem).
        timestamp: Capture time parsed from the filename.
        detections: List of detections found in this frame (may be empty).
    """

    frame_id: str
    timestamp: datetime
    detections: list[Detection]


@dataclass
class Track:
    """A tracked object spanning one or more consecutive frames.

    The tracker matches detections across frames using IoU.  Each match
    increments ``consecutive_hits``; each miss increments
    ``consecutive_misses`` and resets hits.  A track becomes ``confirmed``
    once ``consecutive_hits`` reaches the ``min_consecutive`` threshold.

    Attributes:
        track_id: Unique identifier assigned by the tracker.
        hits: ``(frame_id, detection)`` pairs in temporal order.
        consecutive_hits: Current streak of frames with a matched detection.
        consecutive_misses: Current streak of frames without a match.
        confirmed: Whether the track has met the confirmation threshold.
        confirmed_at_frame: Frame index at which confirmation occurred, or
            ``None`` if the track was never confirmed.
    """

    track_id: int
    hits: list[tuple[str, Detection]] = field(default_factory=list)
    consecutive_hits: int = 0
    consecutive_misses: int = 0
    confirmed: bool = False
    confirmed_at_frame: int | None = None
    mean_confidence: float | None = None
    area_change_ratio: float | None = None


@dataclass
class SequenceResult:
    """Per-sequence evaluation record joining prediction and ground truth.

    Produced by the tracking stage, consumed by the evaluation stage to
    compute sequence-level classification metrics and time-to-detection.

    Attributes:
        sequence_id: Directory name of the sequence.
        is_positive_gt: Ground-truth label (``True`` for wildfire sequences).
        is_positive_pred: Tracker prediction (``True`` if any track was
            confirmed).
        num_frames: Total number of frames in the sequence.
        num_detections_total: Sum of YOLO detections across all frames
            (after confidence filtering).
        num_tracks: Number of tracks created by the tracker.
        confirmed_frame_index: Frame index of the first confirmed track,
            or ``None`` if no track was confirmed.
        confirmed_timestamp: Timestamp of the frame where confirmation
            occurred, or ``None``.
        first_timestamp: Timestamp of the first frame in the sequence,
            or ``None`` if the sequence is empty.
    """

    sequence_id: str
    is_positive_gt: bool
    is_positive_pred: bool
    num_frames: int
    num_detections_total: int
    num_tracks: int
    confirmed_frame_index: int | None = None
    confirmed_timestamp: datetime | None = None
    first_timestamp: datetime | None = None

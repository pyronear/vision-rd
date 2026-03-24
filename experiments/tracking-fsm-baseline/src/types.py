from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Detection:
    """A single YOLO detection in a frame (normalized coords)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float


@dataclass
class FrameResult:
    """All detections for one frame."""

    frame_id: str
    timestamp: datetime
    detections: list[Detection]


@dataclass
class Track:
    """A tracked object across frames."""

    track_id: int
    # (frame_id, detection) pairs in temporal order
    hits: list[tuple[str, Detection]] = field(default_factory=list)
    consecutive_hits: int = 0
    consecutive_misses: int = 0
    confirmed: bool = False
    # Frame index where the track was confirmed (for time-to-detection)
    confirmed_at_frame: int | None = None


@dataclass
class SequenceResult:
    """Tracking result for one sequence."""

    sequence_id: str
    is_positive_gt: bool
    is_positive_pred: bool
    num_frames: int
    num_detections_total: int
    num_tracks: int
    confirmed_frame_index: int | None = None
    confirmed_timestamp: datetime | None = None
    first_timestamp: datetime | None = None

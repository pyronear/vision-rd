"""Shared data types for Pyronear temporal smoke detection experiments."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Frame:
    """A single frame in a temporal sequence.

    Attributes:
        frame_id: Unique identifier, typically the image filename stem.
        image_path: Path to the frame image file.
        timestamp: Capture time, or ``None`` if it cannot be parsed.
    """

    frame_id: str
    image_path: Path
    timestamp: datetime | None = None


@dataclass
class TemporalModelOutput:
    """Output of a temporal model for a single sequence.

    Attributes:
        is_positive: Binary classification decision (``True`` = smoke detected).
        trigger_frame_index: Index of the frame (0-based) where the model
            decided positive, or ``None`` if negative. Time-to-detection
            in frames equals this value for a true positive. Do not
            compute TTD by subtracting frame filename timestamps — they
            are unreliable in the pyro-dataset test set.
        details: Arbitrary model-specific metadata (e.g., tracks, confidence
            scores, attention maps).
    """

    is_positive: bool
    trigger_frame_index: int | None = None
    details: dict = field(default_factory=dict)

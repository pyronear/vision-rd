"""Base class for temporal smoke detection models.

Defines the :class:`TemporalModel` ABC that all experiments implement.
Uses a template-method pattern: :meth:`predict_sequence` wires
:meth:`load_sequence` (overridable) and :meth:`predict` (abstract).
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from pyrocore.types import Frame, TemporalModelOutput

_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")


def _try_parse_timestamp(frame_id: str) -> datetime | None:
    """Attempt to extract a timestamp from a Pyronear-style frame ID.

    Expects the frame ID to end with a timestamp segment matching
    ``YYYY-MM-DDTHH-MM-SS`` (e.g., ``adf_site_999_2023-05-23T17-18-31``).

    Returns:
        Parsed :class:`~datetime.datetime`, or ``None`` if parsing fails.
    """
    match = _TIMESTAMP_RE.search(frame_id)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


class TemporalModel(ABC):
    """Base class for temporal smoke detection models.

    Subclasses **must** implement :meth:`predict`.  They **may** override
    :meth:`load_sequence` to customise how frame image paths are converted
    into :class:`~pyrocore.types.Frame` objects (e.g., to add cached
    detections or parse timestamps with a different naming convention).

    Callers use :meth:`predict_sequence` as the single entry point.

    Example::

        class MyModel(TemporalModel):
            def predict(self, frames: list[Frame]) -> TemporalModelOutput:
                # model logic here
                return TemporalModelOutput(is_positive=True, trigger_frame_index=4)

        model = MyModel()
        output = model.predict_sequence(sorted_frame_paths)
    """

    def load_sequence(self, frames: list[Path]) -> list[Frame]:
        """Load a sequence of frame image paths into :class:`Frame` objects.

        The default implementation builds a :class:`Frame` for each path with:

        - ``frame_id`` set to the filename stem,
        - ``image_path`` set to the input path,
        - ``timestamp`` parsed from the Pyronear filename convention
          (``<prefix>_<YYYY-MM-DDTHH-MM-SS>``), falling back to ``None``.

        Override this method to parse timestamps differently, attach cached
        YOLO detections, or perform other custom loading.

        Args:
            frames: Temporally ordered list of frame image paths.

        Returns:
            List of :class:`Frame` objects in the same order.
        """
        return [
            Frame(
                frame_id=p.stem,
                image_path=p,
                timestamp=_try_parse_timestamp(p.stem),
            )
            for p in frames
        ]

    @abstractmethod
    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        """Run temporal model logic on a loaded sequence.

        This is the method each subclass must implement.

        Args:
            frames: Temporally ordered list of :class:`Frame` objects,
                as returned by :meth:`load_sequence`.

        Returns:
            :class:`TemporalModelOutput` with the classification decision
            and optional timing/details.
        """
        ...

    def predict_sequence(self, frames: list[Path]) -> TemporalModelOutput:
        """Main entry point: load frame images then predict.

        Calls :meth:`load_sequence` followed by :meth:`predict`.  This
        method should generally not be overridden.

        Args:
            frames: Temporally ordered list of frame image paths.

        Returns:
            :class:`TemporalModelOutput` with classification decision and timing.
        """
        loaded = self.load_sequence(frames)
        return self.predict(loaded)

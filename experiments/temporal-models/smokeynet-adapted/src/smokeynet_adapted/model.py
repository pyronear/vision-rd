"""TemporalModel implementation for smokeynet-adapted.

Wires the YOLO companion + tube building + patch cropping + the trained
temporal classifier into the pyrocore :class:`TemporalModel` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pyrocore import Frame, TemporalModel, TemporalModelOutput

from .package import ModelPackage, load_model_package


class SmokeynetTemporalModel(TemporalModel):
    """YOLO companion + tube classifier.

    See ``docs/specs/2026-04-15-temporal-model-protocol-design.md`` for the
    full pipeline description.
    """

    def __init__(
        self,
        *,
        yolo_model: Any,
        classifier: Any,
        config: dict[str, Any],
    ) -> None:
        self._yolo = yolo_model
        self._classifier = classifier
        self._cfg = config

    @classmethod
    def from_package(cls, package_path: Path) -> Self:
        pkg: ModelPackage = load_model_package(package_path)
        return cls(
            yolo_model=pkg.yolo_model,
            classifier=pkg.classifier,
            config=pkg.config,
        )

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        raise NotImplementedError  # implemented in the next task

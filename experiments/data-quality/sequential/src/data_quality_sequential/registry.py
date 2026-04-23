"""Registry of :class:`pyrocore.TemporalModel` implementations.

Single source of truth for the ``--model-type`` flag shared by all scripts.
Each entry is a ``(module_path, class_name)`` pair for a class that exposes
``classmethod from_package(path) -> Self``.
"""

import importlib
from pathlib import Path

from pyrocore import TemporalModel

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "bbox-tube-temporal": (
        "bbox_tube_temporal.model",
        "BboxTubeTemporalModel",
    ),
}


def load_model(model_type: str, package_path: Path) -> TemporalModel:
    """Instantiate a :class:`TemporalModel` from *package_path*.

    Args:
        model_type: Registry key (must be in :data:`MODEL_REGISTRY`).
        package_path: Path to the packaged ``.zip`` produced by the model's
            own packaging stage.

    Raises:
        ValueError: If ``model_type`` is not a registered key.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {model_type!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    module_path, class_name = MODEL_REGISTRY[model_type]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls.from_package(package_path)

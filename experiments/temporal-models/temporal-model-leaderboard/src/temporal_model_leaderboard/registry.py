"""Registry of TemporalModel implementations addressable by short name.

Single source of truth for the ``--model-type`` flag shared by all
leaderboard scripts (``evaluate.py``, ``evaluate_pyro_annotator_export.py``).
"""

import importlib
from pathlib import Path

from pyrocore import TemporalModel

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "fsm-tracking-baseline": (
        "tracking_fsm_baseline.model",
        "FsmTrackingModel",
    ),
    "mtb-change-detection": (
        "mtb_change_detection.model",
        "MtbChangeDetectionModel",
    ),
    "pyro-detector-baseline": (
        "pyro_detector_baseline.model",
        "PyroDetectorModel",
    ),
    "bbox-tube-temporal": (
        "bbox_tube_temporal.model",
        "BboxTubeTemporalModel",
    ),
}


def load_model(model_type: str, package_path: Path) -> TemporalModel:
    """Instantiate a :class:`TemporalModel` from *package_path*.

    Raises ``ValueError`` if *model_type* is not a registered key.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {model_type!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    module_path, class_name = MODEL_REGISTRY[model_type]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls.from_package(package_path)

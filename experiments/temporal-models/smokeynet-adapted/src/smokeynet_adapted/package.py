"""Model packaging: bundle YOLO weights, classifier checkpoint, and config.

The archive is a standard .zip file containing:

- ``manifest.yaml`` — entry point with format version and file pointers.
- ``yolo_weights.pt`` — ultralytics YOLO checkpoint for the companion detector.
- ``classifier.ckpt`` — Lightning checkpoint for ``TemporalSmokeClassifier``.
- ``config.yaml`` — inference config (infer / tubes / model_input / classifier / decision).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
YOLO_WEIGHTS_FILENAME = "yolo_weights.pt"
CLASSIFIER_CKPT_FILENAME = "classifier.ckpt"
CONFIG_FILENAME = "config.yaml"
DEFAULT_EXTRACT_DIR = Path(".cache/smokeynet_model")


@dataclass
class ModelPackage:
    """A loaded model package: classifier, YOLO model, and full config."""

    classifier: Any  # TemporalSmokeClassifier; Any avoids import cycles in this module
    yolo_model: Any  # ultralytics.YOLO; same reason
    config: dict[str, Any]

    @property
    def infer(self) -> dict[str, Any]:
        return self.config["infer"]

    @property
    def tubes(self) -> dict[str, Any]:
        return self.config["tubes"]

    @property
    def model_input(self) -> dict[str, Any]:
        return self.config["model_input"]

    @property
    def classifier_cfg(self) -> dict[str, Any]:
        return self.config["classifier"]

    @property
    def decision(self) -> dict[str, Any]:
        return self.config["decision"]

"""Model packaging: bundle YOLO weights, classifier checkpoint, and config.

The archive is a standard .zip file containing:

- ``manifest.yaml`` — entry point with format version and file pointers.
- ``yolo_weights.pt`` — ultralytics YOLO checkpoint for the companion detector.
- ``classifier.ckpt`` — Lightning checkpoint for ``TemporalSmokeClassifier``.
- ``config.yaml`` — inference config (infer / tubes / model_input / classifier /
  decision).
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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


def build_model_package(
    *,
    yolo_weights_path: Path,
    classifier_ckpt_path: Path,
    config: dict[str, Any],
    variant: str,
    output_path: Path,
) -> Path:
    """Bundle YOLO weights + classifier checkpoint + config into a .zip archive.

    Args:
        yolo_weights_path: Path to the ultralytics YOLO ``.pt`` file.
        classifier_ckpt_path: Path to the Lightning ``.ckpt`` for
            ``TemporalSmokeClassifier``.
        config: Full package config dict (see module docstring for schema).
        variant: Identifier recorded in the manifest (informational).
        output_path: Destination ``.zip`` path.

    Returns:
        The resolved ``output_path``.

    Raises:
        FileNotFoundError: If either input file is missing.
    """
    if not yolo_weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")
    if not classifier_ckpt_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {classifier_ckpt_path}"
        )

    manifest = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "yolo_weights": YOLO_WEIGHTS_FILENAME,
        "classifier_checkpoint": CLASSIFIER_CKPT_FILENAME,
        "config": CONFIG_FILENAME,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest, default_flow_style=False))
        zf.write(yolo_weights_path, YOLO_WEIGHTS_FILENAME)
        zf.write(classifier_ckpt_path, CLASSIFIER_CKPT_FILENAME)
        zf.writestr(CONFIG_FILENAME, yaml.dump(config, default_flow_style=False))
    return output_path.resolve()

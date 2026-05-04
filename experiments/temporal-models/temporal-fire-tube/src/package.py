"""Model packaging: bundle YOLO weights, classifier, and config into an archive.

The archive is a standard ``.zip`` file containing:

- ``manifest.yaml`` -- Entry point with format version and file pointers.
- ``weights.pt`` -- YOLO model checkpoint.
- ``classifier.pkl`` -- Trained Random Forest classifier.
- ``config.yaml`` -- Inference, tube, and classifier parameters.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier

from src.detector import load_model

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
WEIGHTS_FILENAME = "weights.pt"
CLASSIFIER_FILENAME = "classifier.pkl"
CONFIG_FILENAME = "config.yaml"
DEFAULT_EXTRACT_DIR = Path(".cache/model")


@dataclass
class ModelPackage:
    """A loaded model package with YOLO model, RF classifier, and config."""

    model: Any  # ultralytics.YOLO
    classifier: RandomForestClassifier
    config: dict[str, Any]

    @property
    def infer_params(self) -> dict[str, Any]:
        return self.config["infer"]

    @property
    def pad_params(self) -> dict[str, Any]:
        return self.config["pad"]

    @property
    def tube_params(self) -> dict[str, Any]:
        return self.config["tube"]


def build_model_package(
    weights_path: Path,
    classifier_path: Path,
    params: dict[str, Any],
    output_path: Path,
) -> Path:
    """Bundle YOLO weights, RF classifier, and config into a ``.zip`` archive.

    Args:
        weights_path: Path to the YOLO ``.pt`` weights file.
        classifier_path: Path to the trained classifier ``.pkl`` file.
        params: Full ``params.yaml`` dictionary.
        output_path: Destination path for the archive.

    Returns:
        The resolved *output_path*.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found: {classifier_path}")

    config = _build_config(params)
    manifest = {
        "format_version": FORMAT_VERSION,
        "weights": WEIGHTS_FILENAME,
        "classifier": CLASSIFIER_FILENAME,
        "config": CONFIG_FILENAME,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest, default_flow_style=False))
        zf.write(weights_path, WEIGHTS_FILENAME)
        zf.write(classifier_path, CLASSIFIER_FILENAME)
        zf.writestr(CONFIG_FILENAME, yaml.dump(config, default_flow_style=False))

    return output_path.resolve()


def load_model_package(
    package_path: Path,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
) -> ModelPackage:
    """Load a packaged model archive.

    Extracts the YOLO weights and classifier to *extract_dir*, reads the
    manifest, loads models, and parses the config.
    """
    if not package_path.exists():
        raise FileNotFoundError(f"Archive not found: {package_path}")

    with zipfile.ZipFile(package_path, "r") as zf:
        names = zf.namelist()
        if MANIFEST_FILENAME not in names:
            raise KeyError(f"Archive missing {MANIFEST_FILENAME}")

        manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))

        version = manifest.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format_version {version} (expected {FORMAT_VERSION})"
            )

        weights_name = manifest["weights"]
        classifier_name = manifest["classifier"]
        config_name = manifest["config"]

        for name in [weights_name, classifier_name, config_name]:
            if name not in names:
                raise KeyError(f"Archive missing {name}")

        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(weights_name, extract_dir)
        zf.extract(classifier_name, extract_dir)
        config = yaml.safe_load(zf.read(config_name))

    yolo_model = load_model(extract_dir / weights_name)
    classifier = joblib.load(extract_dir / classifier_name)

    return ModelPackage(model=yolo_model, classifier=classifier, config=config)


def _build_config(params: dict[str, Any]) -> dict[str, Any]:
    """Build a package config from the full ``params.yaml`` dictionary."""
    infer = params["infer"]
    pad = params["pad"]
    tube = params["tube"]
    return {
        "infer": {
            "confidence_threshold": infer["confidence_threshold"],
            "iou_nms": infer["iou_nms"],
            "image_size": infer["image_size"],
        },
        "pad": {
            "min_sequence_length": pad["min_sequence_length"],
        },
        "tube": {
            "crop_size": tube["crop_size"],
            "max_tube_length": tube["max_tube_length"],
            "confidence_threshold": tube["confidence_threshold"],
            "max_detection_area": tube["max_detection_area"],
            "iou_threshold": tube["iou_threshold"],
        },
    }

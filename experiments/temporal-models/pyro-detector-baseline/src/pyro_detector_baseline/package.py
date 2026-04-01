"""Model packaging: bundle ONNX weights and predictor config into an archive.

The archive is a standard ``.zip`` file containing:

- ``manifest.yaml`` -- Entry point with format version and file pointers.
- ``best.onnx`` -- ONNX model weights.
- ``config.yaml`` -- Predictor parameters.

This allows shipping the complete pyro-predictor pipeline as one
self-contained artefact.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
WEIGHTS_FILENAME = "best.onnx"
CONFIG_FILENAME = "config.yaml"
DEFAULT_EXTRACT_DIR = Path(".cache/model")


@dataclass
class ModelPackage:
    """A loaded model package with ONNX model path and predictor config."""

    model_path: Path
    config: dict[str, Any]

    @property
    def predict_params(self) -> dict[str, Any]:
        """Predictor constructor parameters."""
        return self.config["predict"]


def build_model_package(
    weights_path: Path,
    params: dict[str, Any],
    output_path: Path,
) -> Path:
    """Bundle ONNX weights and predictor config into a ``.zip`` archive.

    Args:
        weights_path: Path to the ONNX weights file.
        params: Full ``params.yaml`` dictionary (must contain ``predict``).
        output_path: Destination path for the archive.

    Returns:
        The resolved *output_path*.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    config = {"predict": params["predict"]}
    manifest = {
        "format_version": FORMAT_VERSION,
        "weights": WEIGHTS_FILENAME,
        "config": CONFIG_FILENAME,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest, default_flow_style=False))
        zf.write(weights_path, WEIGHTS_FILENAME)
        zf.writestr(CONFIG_FILENAME, yaml.dump(config, default_flow_style=False))

    return output_path.resolve()


def load_model_package(
    package_path: Path,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
) -> ModelPackage:
    """Load a packaged model archive.

    Args:
        package_path: Path to a ``.zip`` archive created by
            :func:`build_model_package`.
        extract_dir: Directory to extract the weights into.

    Returns:
        A :class:`ModelPackage` instance.
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
        config_name = manifest["config"]

        if weights_name not in names:
            raise KeyError(f"Archive missing {weights_name}")
        if config_name not in names:
            raise KeyError(f"Archive missing {config_name}")

        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(weights_name, extract_dir)
        config = yaml.safe_load(zf.read(config_name))

    return ModelPackage(model_path=extract_dir / weights_name, config=config)

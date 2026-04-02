"""Model packaging: bundle YOLO weights, network weights, and config.

The archive is a ``.zip`` file containing:

- ``manifest.yaml`` -- Format version and file pointers.
- ``yolo_weights.pt`` -- YOLO model checkpoint.
- ``net_weights.pt`` -- SmokeyNetAdapted state dict (LSTM + ViT + heads).
- ``config.yaml`` -- Inference and architecture parameters.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .detector import load_model as load_yolo_model
from .net import SmokeyNetAdapted

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
YOLO_WEIGHTS_FILENAME = "yolo_weights.pt"
NET_WEIGHTS_FILENAME = "net_weights.pt"
CONFIG_FILENAME = "config.yaml"
DEFAULT_EXTRACT_DIR = Path(".cache/smokeynet_model")


@dataclass
class ModelPackage:
    """A loaded model package with YOLO model, network, and config."""

    yolo_model: Any
    net: SmokeyNetAdapted
    config: dict[str, Any]

    @property
    def infer_params(self) -> dict[str, Any]:
        return self.config["infer"]

    @property
    def extract_params(self) -> dict[str, Any]:
        return self.config["extract"]

    @property
    def tubes_params(self) -> dict[str, Any]:
        return self.config["tubes"]

    @property
    def classification_threshold(self) -> float:
        return self.config.get("classification_threshold", 0.5)


def build_model_package(
    yolo_weights_path: Path,
    net_weights_path: Path,
    params: dict[str, Any],
    output_path: Path,
) -> Path:
    """Bundle YOLO weights, network weights, and config into a ``.zip``.

    Args:
        yolo_weights_path: Path to the YOLO ``.pt`` weights.
        net_weights_path: Path to the SmokeyNetAdapted checkpoint.
        params: Full ``params.yaml`` dictionary.
        output_path: Destination path for the archive.

    Returns:
        Resolved *output_path*.
    """
    if not yolo_weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")
    if not net_weights_path.exists():
        raise FileNotFoundError(f"Net weights not found: {net_weights_path}")

    config = _build_config(params)
    manifest = {
        "format_version": FORMAT_VERSION,
        "yolo_weights": YOLO_WEIGHTS_FILENAME,
        "net_weights": NET_WEIGHTS_FILENAME,
        "config": CONFIG_FILENAME,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            MANIFEST_FILENAME,
            yaml.dump(manifest, default_flow_style=False),
        )
        zf.write(yolo_weights_path, YOLO_WEIGHTS_FILENAME)
        zf.write(net_weights_path, NET_WEIGHTS_FILENAME)
        zf.writestr(
            CONFIG_FILENAME,
            yaml.dump(config, default_flow_style=False),
        )

    return output_path.resolve()


def load_model_package(
    package_path: Path,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
) -> ModelPackage:
    """Load a packaged model archive.

    Args:
        package_path: Path to the ``.zip`` archive.
        extract_dir: Directory to extract weights into.

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

        config = yaml.safe_load(zf.read(manifest["config"]))

        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(manifest["yolo_weights"], extract_dir)
        zf.extract(manifest["net_weights"], extract_dir)

    # Load YOLO model
    yolo_weights = extract_dir / manifest["yolo_weights"]
    yolo_model = load_yolo_model(yolo_weights)

    # Build and load SmokeyNetAdapted
    train_cfg = config["train"]
    net = SmokeyNetAdapted(
        d_model=train_cfg["d_model"],
        lstm_layers=train_cfg["lstm_layers"],
        spatial_layers=train_cfg["spatial_layers"],
        spatial_heads=train_cfg["spatial_heads"],
    )
    net_weights = extract_dir / manifest["net_weights"]
    state_dict = torch.load(net_weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    return ModelPackage(yolo_model=yolo_model, net=net, config=config)


def _build_config(params: dict[str, Any]) -> dict[str, Any]:
    """Extract inference-relevant config from full params."""
    return {
        "infer": params["infer"],
        "extract": params["extract"],
        "tubes": params["tubes"],
        "train": {
            "d_model": params["train"]["d_model"],
            "lstm_layers": params["train"]["lstm_layers"],
            "spatial_layers": params["train"]["spatial_layers"],
            "spatial_heads": params["train"]["spatial_heads"],
        },
        "classification_threshold": params["train"]["classification_threshold"],
    }

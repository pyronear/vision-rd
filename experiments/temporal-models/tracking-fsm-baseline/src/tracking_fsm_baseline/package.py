"""Model packaging: bundle YOLO weights and tracking config into a single archive.

The archive is a standard ``.zip`` file containing:

- ``manifest.yaml`` -- Entry point with format version and file pointers.
- ``weights.pt`` -- YOLO model checkpoint.
- ``config.yaml`` -- Inference, prefilter, and tracker parameters.

This allows shipping the complete smoke-detection pipeline (detector + FSM
tracker settings) as one self-contained artefact.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .detector import load_model
from .tracker import SimpleTracker

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
WEIGHTS_FILENAME = "weights.pt"
CONFIG_FILENAME = "config.yaml"
DEFAULT_EXTRACT_DIR = Path(".cache/model")


@dataclass
class ModelPackage:
    """A loaded model package with YOLO model and pipeline config.

    Example::

        pkg = load_model_package(Path("model.zip"))
        tracker = pkg.create_tracker()
    """

    model: Any  # ultralytics.YOLO — typed as Any to avoid import at module level
    config: dict[str, Any]

    @property
    def infer_params(self) -> dict[str, Any]:
        """YOLO ``predict()`` parameters."""
        return self.config["infer"]

    @property
    def pad_params(self) -> dict[str, Any]:
        """Sequence padding parameters."""
        return self.config["pad"]

    @property
    def prefilter_params(self) -> dict[str, Any]:
        """Detection pre-filter parameters (applied before the tracker)."""
        return self.config["prefilter"]

    @property
    def tracker_params(self) -> dict[str, Any]:
        """``SimpleTracker.__init__`` keyword arguments."""
        return self.config["tracker"]

    def create_tracker(self) -> SimpleTracker:
        """Instantiate a :class:`SimpleTracker` from the package config."""
        return SimpleTracker(**self.tracker_params)


def build_model_package(
    weights_path: Path,
    params: dict[str, Any],
    output_path: Path,
) -> Path:
    """Bundle YOLO weights and tracking config into a ``.zip`` archive.

    Args:
        weights_path: Path to the YOLO ``.pt`` weights file.
        params: Full ``params.yaml`` dictionary (must contain ``infer`` and
            ``track`` sections).
        output_path: Destination path for the archive.

    Returns:
        The resolved *output_path*.

    Raises:
        FileNotFoundError: If *weights_path* does not exist.
        KeyError: If *params* is missing required sections.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    config = _build_config(params)
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

    Extracts the YOLO weights to *extract_dir*, reads the manifest,
    loads the model, and parses the config.

    Args:
        package_path: Path to a ``.zip`` archive created by
            :func:`build_model_package`.
        extract_dir: Directory to extract the weights into.  Defaults to
            ``.cache/model/`` relative to the current working directory.

    Returns:
        A :class:`ModelPackage` instance.

    Raises:
        FileNotFoundError: If *package_path* does not exist.
        KeyError: If the archive is missing expected entries.
        ValueError: If the manifest ``format_version`` is unsupported.
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

    weights_on_disk = extract_dir / weights_name
    model = load_model(weights_on_disk)

    return ModelPackage(model=model, config=config)


def _build_config(params: dict[str, Any]) -> dict[str, Any]:
    """Build a package config from the full ``params.yaml`` dictionary.

    Extracts only the fields relevant at inference time and reshapes them
    into the package config schema with ``infer``, ``prefilter``, ``pad``,
    and ``tracker`` sections.
    """
    infer = params["infer"]
    pad = params["pad"]
    track = params["track"]
    return {
        "infer": {
            "confidence_threshold": infer["confidence_threshold"],
            "iou_nms": infer["iou_nms"],
            "image_size": infer["image_size"],
        },
        "pad": {
            "min_sequence_length": pad["min_sequence_length"],
        },
        "prefilter": {
            "confidence_threshold": track["confidence_threshold"],
            "max_detection_area": track["max_detection_area"],
        },
        "tracker": {
            "iou_threshold": track["iou_threshold"],
            "min_consecutive": track["min_consecutive"],
            "max_misses": track["max_misses"],
            "use_confidence_filter": track["use_confidence_filter"],
            "min_mean_confidence": track["min_mean_confidence"],
            "use_area_change_filter": track["use_area_change_filter"],
            "min_area_change": track["min_area_change"],
        },
    }

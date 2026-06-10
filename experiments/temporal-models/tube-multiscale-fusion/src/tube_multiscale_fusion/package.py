"""Model packaging: bundle YOLO weights, classifier checkpoint, and config.

Mirrors ``bbox_tube_temporal.package`` (the sibling experiment) so the
two-branch model plugs into the same serving / leaderboard path. The archive
is a standard ``.zip`` containing:

- ``manifest.yaml`` — entry point with format version and file pointers.
- ``yolo_weights.pt`` — ultralytics YOLO checkpoint for the companion detector.
- ``classifier.ckpt`` — Lightning checkpoint for ``LitTubeMultiscaleClassifier``.
- ``config.yaml`` — inference config (infer / tubes / model_input / classifier /
  decision).

Unlike ``bbox_tube_temporal`` this experiment ships only the ``max_logit``
decision rule, so there is no bundled logistic calibrator.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .lit_module import LitTubeMultiscaleClassifier

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
YOLO_WEIGHTS_FILENAME = "yolo_weights.pt"
CLASSIFIER_CKPT_FILENAME = "classifier.ckpt"
CONFIG_FILENAME = "config.yaml"
DEFAULT_EXTRACT_DIR = Path(".cache/tube_multiscale_fusion_model")


@dataclass
class ModelPackage:
    """A loaded model package: classifier, YOLO model, and full config."""

    classifier: Any  # LitTubeMultiscaleClassifier; Any avoids an import cycle
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
            ``LitTubeMultiscaleClassifier``.
        config: Full package config dict (see module docstring for schema).
        variant: Identifier recorded in the manifest (informational).
        output_path: Destination ``.zip`` path.

    Returns:
        The resolved ``output_path``.

    Raises:
        FileNotFoundError: If either input file is missing.
    """
    import zipfile  # noqa: PLC0415 — keep the heavy stdlib import local to the writer

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


def _load_yolo(weights_path: Path) -> Any:
    """Thin wrapper around ultralytics.YOLO.

    The ``ultralytics`` import is deliberately inside the function body so
    tests can patch ``_load_yolo`` without triggering the heavy import chain
    (mirrors ``bbox_tube_temporal.package._load_yolo``). ``ultralytics`` is
    pulled in transitively via ``bbox-tube-temporal-core``.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO(str(weights_path))


def _load_classifier(ckpt_path: Path) -> LitTubeMultiscaleClassifier:
    """Load a ``LitTubeMultiscaleClassifier`` from a Lightning checkpoint.

    ``pretrained=False`` is forced so timm does not re-download backbone
    weights — the trained weights already live in the checkpoint.
    """
    model = LitTubeMultiscaleClassifier.load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu",
        pretrained=False,
    )
    model.eval()
    return model


def load_model_package(
    package_path: Path,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
) -> ModelPackage:
    """Load a packaged model archive.

    Args:
        package_path: Path to a ``.zip`` built by :func:`build_model_package`.
        extract_dir: Where to extract YOLO weights and classifier ckpt.

    Raises:
        FileNotFoundError: if ``package_path`` does not exist.
        KeyError: if the archive is missing expected entries.
        ValueError: if ``format_version`` is unsupported.
    """
    import zipfile  # noqa: PLC0415

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

        yolo_name = manifest["yolo_weights"]
        ckpt_name = manifest["classifier_checkpoint"]
        config_name = manifest["config"]
        for n in (yolo_name, ckpt_name, config_name):
            if n not in names:
                raise KeyError(f"Archive missing {n}")

        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(yolo_name, extract_dir)
        zf.extract(ckpt_name, extract_dir)
        config = yaml.safe_load(zf.read(config_name))

    yolo_model = _load_yolo(extract_dir / yolo_name)
    classifier = _load_classifier(extract_dir / ckpt_name)
    return ModelPackage(classifier=classifier, yolo_model=yolo_model, config=config)

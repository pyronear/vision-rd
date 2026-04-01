"""Bundle ONNX weights and predictor config into a single model archive.

Reads the experiment ``params.yaml``, extracts the predict parameters,
and packages them together with the ONNX weights into a ``.zip``
archive that can be deployed independently.

Usage:
    uv run python scripts/package.py \
        --model-dir data/01_raw/models \
        --params-path params.yaml \
        --output-path data/06_models/model.zip
"""

import argparse
import logging
from pathlib import Path

import yaml

from pyro_detector_baseline.package import build_model_package

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _find_onnx_model(model_dir: Path) -> Path:
    """Find the ONNX model file in a directory, skipping macOS resource forks."""
    onnx_files = [f for f in model_dir.glob("**/*.onnx") if not f.name.startswith("._")]
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX model found in {model_dir}")
    return onnx_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle ONNX weights and predictor config into a model archive."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to directory containing the ONNX model.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        required=True,
        help="Path to params.yaml.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination path for the model archive (.zip).",
    )
    args = parser.parse_args()

    weights_path = _find_onnx_model(args.model_dir)
    logger.info("Using weights: %s", weights_path)

    params = yaml.safe_load(args.params_path.read_text())
    output = build_model_package(weights_path, params, args.output_path)

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("Model archive saved to %s (%.1f MB)", output, size_mb)


if __name__ == "__main__":
    main()

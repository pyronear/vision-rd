"""Bundle YOLO weights and MTB+tracking config into a single model archive.

Reads the experiment ``params.yaml``, extracts the inference, change detection,
and tracking parameters, and packages them together with the YOLO weights into
a ``.zip`` archive that can be deployed independently.

Usage:
    uv run python scripts/package.py \
        --weights-path data/01_raw/models/best.pt \
        --params-path params.yaml \
        --output-path data/06_models/model.zip
"""

import argparse
import logging
from pathlib import Path

import yaml

from mtb_change_detection.package import build_model_package

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle YOLO weights and MTB+tracking config into a model archive."
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        required=True,
        help="Path to YOLO .pt weights file.",
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

    params = yaml.safe_load(args.params_path.read_text())
    output = build_model_package(args.weights_path, params, args.output_path)

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("Model archive saved to %s (%.1f MB)", output, size_mb)


if __name__ == "__main__":
    main()

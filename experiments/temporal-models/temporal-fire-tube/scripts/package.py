"""Bundle YOLO weights, trained classifier, and config into a model archive.

Usage:
    uv run python scripts/package.py \
        --weights-path data/01_raw/models/yolo11s_mighty-mongoose_v5.1.0.pt \
        --classifier-path data/06_models/classifier.pkl \
        --params-path params.yaml \
        --output-path data/06_models/model.zip
"""

import argparse
import logging
from pathlib import Path

import yaml

from src.package import build_model_package

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle YOLO weights and classifier into a model archive."
    )
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--classifier-path", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    params = yaml.safe_load(args.params_path.read_text())
    output = build_model_package(
        args.weights_path, args.classifier_path, params, args.output_path
    )

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("Model archive saved to %s (%.1f MB)", output, size_mb)


if __name__ == "__main__":
    main()

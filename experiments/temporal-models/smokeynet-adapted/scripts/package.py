"""Bundle YOLO weights, network weights, and config into a model archive.

Usage:
    uv run python scripts/package.py \
        --yolo-weights-path data/01_raw/models/yolo11s_nimble-narwhal_v6.0.0.pt \
        --net-weights-path data/06_models/best_checkpoint.pt \
        --params-path params.yaml \
        --output-path data/06_models/model.zip
"""

import argparse
import logging
from pathlib import Path

import yaml

from smokeynet_adapted.package import build_model_package

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle model into a deployable archive."
    )
    parser.add_argument("--yolo-weights-path", type=Path, required=True)
    parser.add_argument("--net-weights-path", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    params = yaml.safe_load(args.params_path.read_text())
    output = build_model_package(
        args.yolo_weights_path,
        args.net_weights_path,
        params,
        args.output_path,
    )

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("Model archive saved to %s (%.1f MB)", output, size_mb)


if __name__ == "__main__":
    main()

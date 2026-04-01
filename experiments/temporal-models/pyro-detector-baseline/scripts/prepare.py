"""Download ONNX model weights from HuggingFace Hub.

Fetches the ONNX model archive from a HuggingFace repository, extracts
it, and saves it to the local data directory.  Skips the download if
the extracted model already exists.

Usage:
    uv run python scripts/prepare.py \
        --model-repo pyronear/yolo11s_mighty-mongoose_v5.1.0 \
        --model-filename onnx_cpu_yolo11s_mighty-mongoose_v5.1.0.tar.gz \
        --output-model-dir data/01_raw/models
"""

import argparse
import logging
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ONNX model.")
    parser.add_argument(
        "--model-repo",
        type=str,
        required=True,
        help="HuggingFace model repo ID.",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="Model archive filename in the HuggingFace repo.",
    )
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        required=True,
        help="Output directory for model weights.",
    )
    args = parser.parse_args()

    args.output_model_dir.mkdir(parents=True, exist_ok=True)

    # Check if archive already downloaded
    archive_path = args.output_model_dir / args.model_filename
    if not archive_path.exists():
        logger.info("Downloading %s from %s...", args.model_filename, args.model_repo)
        hf_hub_download(
            repo_id=args.model_repo,
            filename=args.model_filename,
            local_dir=args.output_model_dir,
        )
        logger.info("Archive saved to %s", archive_path)
    else:
        logger.info("Archive already exists at %s, skipping download.", archive_path)

    # Extract tar.gz if not already extracted
    onnx_files = list(args.output_model_dir.glob("**/*.onnx"))
    if not onnx_files:
        logger.info("Extracting %s...", archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=args.output_model_dir)
        onnx_files = list(args.output_model_dir.glob("**/*.onnx"))
        logger.info("Extracted ONNX model(s): %s", [str(f) for f in onnx_files])
    else:
        logger.info("ONNX model already extracted: %s", [str(f) for f in onnx_files])


if __name__ == "__main__":
    main()

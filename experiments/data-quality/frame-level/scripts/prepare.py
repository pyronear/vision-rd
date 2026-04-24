"""Download a YOLO .pt checkpoint from Hugging Face.

Fetches a single file from a HuggingFace repo via hf_hub_download, copies
it to the requested output path, and exits. Idempotent — skips the
download if the output already exists.

Usage::

    uv run python scripts/prepare.py \\
        --hf-repo pyronear/yolo11s_nimble-narwhal_v6.0.0 \\
        --hf-filename best.pt \\
        --output data/01_raw/models/yolo11s-nimble-narwhal.pt
"""

import argparse
import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", required=True, type=str)
    parser.add_argument("--hf-filename", required=True, type=str)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.is_file():
        logger.info("Model already at %s — skipping download.", args.output)
        return

    logger.info("Downloading %s from %s ...", args.hf_filename, args.hf_repo)
    cached_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename)
    shutil.copy(cached_path, args.output)
    logger.info("Copied to %s", args.output)


if __name__ == "__main__":
    main()

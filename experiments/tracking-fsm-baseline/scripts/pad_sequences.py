"""Pad short inference sequences by symmetrically repeating boundary frames.

Reads per-sequence inference JSONs, pads any sequence shorter than
``--min-sequence-length`` by alternately prepending the first frame and
appending the last frame, and writes the (possibly padded) results to an
output directory. Sequences that already meet the minimum length are
copied unchanged.

Usage:
    uv run python scripts/pad_sequences.py \
        --infer-dir data/02_intermediate/train \
        --output-dir data/03_primary/train \
        --min-sequence-length 10
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from src.detector import load_inference_results, save_inference_results
from src.tracker import pad_sequence

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pad short sequences by repeating the last frame.",
    )
    parser.add_argument(
        "--infer-dir",
        type=Path,
        required=True,
        help="Path to inference results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for padded inference results.",
    )
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        required=True,
        help="Minimum number of frames per sequence.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Found %d inference files.", len(infer_files))

    n_padded = 0
    for infer_path in tqdm(infer_files, desc="Padding"):
        frames = load_inference_results(infer_path)
        original_len = len(frames)
        frames = pad_sequence(frames, args.min_sequence_length)
        if len(frames) > original_len:
            n_padded += 1
        save_inference_results(frames, args.output_dir / infer_path.name)

    logger.info(
        "Padded %d/%d sequences (min_length=%d). Saved to %s",
        n_padded,
        len(infer_files),
        args.min_sequence_length,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

"""Build a pool of hard-negative distractor patches from false-positive frames.

pyro-dataset's fp/ frames are whole smoke-like NON-smoke scenes (clouds, fog,
glare, haze) that a prior detector fired on. Their per-box "detector-mistake"
labels are dropped in our flat import, but the entire frame is a false-positive
context, so any crop of it is a valid hard negative. This extracts several random
patches per fp frame (at native resolution, smoke-box-like scales) into a pool
that the ``FPInject`` transform pastes onto training tiles WITHOUT adding a box —
teaching the model to ignore smoke-like distractors.

    uv run python scripts/build_fp_crop_pool.py \
        --fp-dir data/01_raw/fp_frames --output-dir data/05_model_input/fp_crops \
        --patches-per-frame 6 --min-size 48 --max-size 160 --seed 42
"""

import argparse
import logging
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_split(
    src: Path, dst: Path, n: int, lo: int, hi: int, rng: random.Random
) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in tqdm(sorted(src.glob("*.jpg")), desc=f"fp-crops {src.name}"):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            for i in range(n):
                s = rng.randint(lo, min(hi, w, h))
                x = rng.randint(0, w - s)
                y = rng.randint(0, h - s)
                patch = im.crop((x, y, x + s, y + s))
                patch.save(dst / f"{img_path.stem}_{i}.jpg", quality=95)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fp hard-negative crop pool.")
    parser.add_argument("--fp-dir", type=Path, required=True)  # has train/ val/ subdirs
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patches-per-frame", type=int, default=6)
    parser.add_argument("--min-size", type=int, default=48)
    parser.add_argument("--max-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for split in ("train", "val"):
        src = args.fp_dir / split
        if not src.is_dir():
            continue
        n = _build_split(
            src,
            args.output_dir / split,
            args.patches_per_frame,
            args.min_size,
            args.max_size,
            rng,
        )
        logger.info("%s -> %s: %d distractor patches", src, args.output_dir / split, n)


if __name__ == "__main__":
    main()

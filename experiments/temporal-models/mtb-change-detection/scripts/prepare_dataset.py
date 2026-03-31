"""Copy and truncate the pyro-dataset for this experiment.

Copies sequences from the source pyro-dataset into data/01_raw/datasets/,
preserving the {wildfire,fp}/<sequence>/ structure and truncating each
sequence to a maximum number of frames (sorted by timestamp, keeping the
earliest).
"""

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to sequential_train_val/ in pyro-dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output path (e.g. data/01_raw/datasets)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum number of frames per sequence (default: 20)",
    )
    return parser.parse_args()


def truncate_sequence(src_seq: Path, dst_seq: Path, max_frames: int) -> None:
    """Copy a sequence directory, truncating to max_frames."""
    src_images = sorted((src_seq / "images").glob("*.jpg"))
    src_labels = src_seq / "labels"

    # Keep first max_frames (earliest by filename/timestamp)
    kept_images = src_images[:max_frames]
    kept_stems = {p.stem for p in kept_images}

    # Copy images
    dst_images = dst_seq / "images"
    dst_images.mkdir(parents=True, exist_ok=True)
    for img in kept_images:
        shutil.copy2(img, dst_images / img.name)

    # Copy matching labels
    dst_labels = dst_seq / "labels"
    dst_labels.mkdir(parents=True, exist_ok=True)
    if src_labels.is_dir():
        for label_file in sorted(src_labels.glob("*.txt")):
            if label_file.stem in kept_stems:
                shutil.copy2(label_file, dst_labels / label_file.name)


def main() -> None:
    args = parse_args()

    for split in ("train", "val"):
        src_split = args.source_dir / split
        if not src_split.is_dir():
            print(f"Skipping {split}: {src_split} not found")
            continue

        for category in ("wildfire", "fp"):
            src_cat = src_split / category
            if not src_cat.is_dir():
                print(f"Skipping {split}/{category}: not found")
                continue

            sequences = sorted(d for d in src_cat.iterdir() if d.is_dir())
            print(f"{split}/{category}: {len(sequences)} sequences")

            for seq_dir in sequences:
                dst_seq = args.output_dir / split / category / seq_dir.name
                if dst_seq.exists():
                    continue
                truncate_sequence(seq_dir, dst_seq, args.max_frames)

    print("Done.")


if __name__ == "__main__":
    main()

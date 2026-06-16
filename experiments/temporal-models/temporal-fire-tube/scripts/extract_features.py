"""Extract tabular features from fire-tubes.

Loads tube crops and metadata from disk, computes temporal features for
each tube, and saves feature vectors + labels as .npz files.

Usage:
    uv run python scripts/extract_features.py \
        --tube-dir data/04_feature/train \
        --output-dir data/05_model_input/train
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.features import extract_tabular_features
from src.types import Detection, FireTube, TubeCrop

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from fire-tubes.")
    parser.add_argument("--tube-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(d for d in args.tube_dir.iterdir() if d.is_dir())
    logger.info("Found %d sequence directories.", len(seq_dirs))

    total_tubes = 0
    for seq_dir in tqdm(seq_dirs, desc="Extracting features"):
        seq_id = seq_dir.name
        output_path = args.output_dir / f"{seq_id}.npz"

        tubes = _load_tubes(seq_dir)
        if not tubes:
            # Save empty arrays for consistency
            np.savez(
                output_path,
                features=np.zeros((0, 24), dtype=np.float64),
                labels=np.zeros(0, dtype=np.int64),
                tube_ids=np.zeros(0, dtype=np.int64),
            )
            continue

        features = []
        labels = []
        tube_ids = []
        for tube in tubes:
            feat = extract_tabular_features(tube)
            features.append(feat)
            labels.append(int(tube.label) if tube.label is not None else 0)
            tube_ids.append(tube.tube_id)

        np.savez(
            output_path,
            features=np.array(features),
            labels=np.array(labels, dtype=np.int64),
            tube_ids=np.array(tube_ids, dtype=np.int64),
        )
        total_tubes += len(tubes)

    logger.info(
        "Extracted features for %d tubes. Saved to %s",
        total_tubes,
        args.output_dir,
    )


def _load_tubes(seq_dir: Path) -> list[FireTube]:
    """Load tubes from a sequence directory (crops.npz + metadata.json)."""
    meta_path = seq_dir / "metadata.json"
    if not meta_path.exists():
        return []

    metadata = json.loads(meta_path.read_text())
    if not metadata:
        return []

    crops_path = seq_dir / "crops.npz"
    crops_array = np.load(crops_path)["crops"] if crops_path.exists() else None

    tubes = []
    for tube_meta in metadata:
        crops = []
        for crop_meta in tube_meta["crops"]:
            det_data = crop_meta["detection"]
            detection = Detection(
                class_id=det_data["class_id"],
                cx=det_data["cx"],
                cy=det_data["cy"],
                w=det_data["w"],
                h=det_data["h"],
                confidence=det_data["confidence"],
            )
            image = (
                crops_array[crop_meta["crop_idx"]]
                if crops_array is not None
                else np.zeros((64, 64, 3), dtype=np.uint8)
            )
            crops.append(
                TubeCrop(
                    frame_id=crop_meta["frame_id"],
                    timestamp=datetime.fromisoformat(crop_meta["timestamp"]),
                    image=image,
                    detection=detection,
                )
            )
        tubes.append(
            FireTube(
                tube_id=tube_meta["tube_id"],
                sequence_id=tube_meta["sequence_id"],
                crops=crops,
                label=tube_meta.get("label"),
            )
        )
    return tubes


if __name__ == "__main__":
    main()

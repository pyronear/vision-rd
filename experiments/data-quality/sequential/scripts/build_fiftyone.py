"""Build FiftyOne FP and FN review datasets from a review manifest.

One FiftyOne dataset per error kind, per (model, split). Each sample is one
frame; samples are grouped by a ``sequence_name`` field so reviewers can
browse sequence by sequence.

Usage::

    uv run --group explore python scripts/build_fiftyone.py \\
        --manifest-path data/08_reporting/<model>/<split>/review_manifest.json \\
        --output-dir data/fiftyone/<model>/<split>
"""

import argparse
import json
import logging
from pathlib import Path

import fiftyone as fo

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _frames_for_sequence(split_dir: Path, ground_truth: bool, name: str) -> list[Path]:
    bucket = "wildfire" if ground_truth else "fp"
    images_dir = split_dir / bucket / name / "images"
    if not images_dir.is_dir():
        return []
    return sorted(images_dir.glob("*.jpg"))


def _build_dataset(
    dataset_name: str,
    split_dir: Path,
    entries: list[dict],
) -> int:
    """Create (or overwrite) a FiftyOne dataset; return sample count."""
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name, persistent=True)

    samples: list[fo.Sample] = []
    for entry in entries:
        frames = _frames_for_sequence(
            split_dir=split_dir,
            ground_truth=entry["ground_truth"],
            name=entry["sequence_name"],
        )
        for frame_idx, frame_path in enumerate(frames):
            samples.append(
                fo.Sample(
                    filepath=str(frame_path),
                    sequence_name=entry["sequence_name"],
                    split=entry["split"],
                    model_name=entry["model_name"],
                    ground_truth=entry["ground_truth"],
                    predicted=entry["predicted"],
                    trigger_frame_index=entry["trigger_frame_index"],
                    frame_index=frame_idx,
                )
            )
    if samples:
        dataset.add_samples(samples)
    return len(samples)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest_path.read_text())
    model_name = manifest["model_name"]
    split = manifest["split"]
    split_dir = Path(manifest["split_dir"])

    fp_dataset_name = f"dq-seq_{model_name}_{split}_fp"
    fn_dataset_name = f"dq-seq_{model_name}_{split}_fn"

    fp_samples = _build_dataset(fp_dataset_name, split_dir, manifest["fp"])
    fn_samples = _build_dataset(fn_dataset_name, split_dir, manifest["fn"])

    sentinel = {
        "model_name": model_name,
        "split": split,
        "fp_dataset": fp_dataset_name,
        "fp_sequences": len(manifest["fp"]),
        "fp_samples": fp_samples,
        "fn_dataset": fn_dataset_name,
        "fn_sequences": len(manifest["fn"]),
        "fn_samples": fn_samples,
    }
    (args.output_dir / "datasets.json").write_text(json.dumps(sentinel, indent=2))

    logger.info(
        "Built FiftyOne datasets %s (%d samples) and %s (%d samples)",
        fp_dataset_name,
        fp_samples,
        fn_dataset_name,
        fn_samples,
    )


if __name__ == "__main__":
    main()

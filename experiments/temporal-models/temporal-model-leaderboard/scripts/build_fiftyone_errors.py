"""Build a FiftyOne dataset of pyro-annotator misclassifications.

Consumes the per-sequence JSONs written by
``scripts/evaluate_pyro_annotator_export.py``, filters to false negatives
(smoke predicted fp) and false positives (fp predicted smoke), and emits a
FiftyOne dataset with one sample per frame.

Each sample carries:

* ``yolo_prior_detections`` (purple) -- per-frame YOLO boxes parsed from the
  sibling ``labels_predictions/<stem>.txt``. These are the edge detector's
  outputs shipped with the dataset, not human GT (the dataset has none).
* ``tube_detections`` (red, ``index=tube_id`` for per-tube coloring) -- per-frame
  boxes from the model's kept tubes, interpolated where gaps were filled. Only
  populated when the underlying model carries tube details (bbox-tube-temporal).

Usage::

    uv run --group explore python scripts/build_fiftyone_errors.py \\
        --predictions-dir data/08_reporting/pyro_annotator/<model-name> \\
        --export-dir data/01_raw/pyro_annotator_exports/sdis-77-new_model/sdis-77
"""

import argparse
import json
import logging
from pathlib import Path

import fiftyone as fo
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Map a prediction record's ``label`` (smoke/fp) to the on-disk parent folder.
LABEL_FOLDER = {"smoke": "smoke", "fp": "false_positive"}

DATASET_NAME_PREFIX = "leaderboard-pyro-annotator"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Output directory from evaluate_pyro_annotator_export.py.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Root of the pyro-annotator export used for evaluation.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "FiftyOne dataset name. Defaults to "
            "'leaderboard-pyro-annotator-<model_name>-errors' where "
            "<model_name> is read from metrics.json."
        ),
    )
    return parser.parse_args()


def _classify(label: str, is_positive: bool) -> str | None:
    """Return ``"false_negative"`` / ``"false_positive"`` or ``None`` for correct."""
    if label == "smoke" and not is_positive:
        return "false_negative"
    if label == "fp" and is_positive:
        return "false_positive"
    return None


def _parse_yolo_labels(path: Path) -> list[tuple[float, float, float, float, float]]:
    """Parse YOLO ``class cx cy w h [conf]`` lines.

    Returns a list of ``(cx, cy, w, h, confidence)`` tuples; confidence
    defaults to ``1.0`` when absent.
    """
    if not path.is_file():
        return []
    boxes: list[tuple[float, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, w, h, *rest = parts
        conf = float(rest[0]) if rest else 1.0
        boxes.append((float(cx), float(cy), float(w), float(h), conf))
    return boxes


def _yolo_to_fo_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Convert YOLO center-based to FiftyOne top-left (x, y, w, h)."""
    return [cx - w / 2, cy - h / 2, w, h]


def _prior_detections(image_path: Path) -> list[fo.Detection]:
    """YOLO-prior boxes for a frame, read from the sibling ``labels_predictions``."""
    labels_txt = (
        image_path.parent.parent / "labels_predictions" / f"{image_path.stem}.txt"
    )
    return [
        fo.Detection(
            label="smoke_prior",
            bounding_box=_yolo_to_fo_bbox(cx, cy, w, h),
            confidence=conf,
        )
        for cx, cy, w, h, conf in _parse_yolo_labels(labels_txt)
    ]


def _tube_detections_for_frame(
    kept_tubes: list[dict], frame_idx: int
) -> list[fo.Detection]:
    """Emit one ``fo.Detection`` per tube that has a bbox at ``frame_idx``.

    ``index=tube_id`` gives consistent per-tube coloring across frames.
    """
    dets: list[fo.Detection] = []
    for tube in kept_tubes:
        tube_id = int(tube["tube_id"])
        for entry in tube.get("entries", []):
            if entry["frame_idx"] != frame_idx or entry.get("bbox") is None:
                continue
            cx, cy, w, h = entry["bbox"]
            dets.append(
                fo.Detection(
                    label=f"tube_{tube_id}",
                    bounding_box=_yolo_to_fo_bbox(cx, cy, w, h),
                    confidence=float(tube["logit"]),
                    index=tube_id,
                )
            )
            break
    return dets


def _build_samples_for_sequence(
    *,
    record: dict,
    seq_dir: Path,
    category: str,
) -> list[fo.Sample]:
    """Build per-frame FiftyOne samples for one misclassified sequence."""
    image_paths = sorted((seq_dir / "images").glob("*.jpg"))
    if not image_paths:
        logger.warning("No images found for %s, skipping.", record["sequence_id"])
        return []

    kept_tubes = record.get("kept_tubes", [])

    samples: list[fo.Sample] = []
    for idx, img_path in enumerate(image_paths):
        sample = fo.Sample(filepath=str(img_path.resolve()))
        sample.tags = [category]
        sample["category"] = category
        sample["sequence_id"] = record["sequence_id"]
        sample["subcategory"] = record.get("subcategory")
        sample["frame_index"] = idx
        sample["ground_truth"] = fo.Classification(
            label="smoke" if record["label"] == "smoke" else "no_smoke"
        )
        sample["prediction"] = fo.Classification(
            label="smoke" if record["is_positive"] else "no_smoke"
        )
        sample["score"] = record.get("score")
        sample["trigger_frame_index"] = record.get("trigger_frame_index")
        sample["ttd_frames"] = record.get("ttd_frames")

        priors = _prior_detections(img_path)
        if priors:
            sample["yolo_prior_detections"] = fo.Detections(detections=priors)

        tubes = _tube_detections_for_frame(kept_tubes, idx)
        if tubes:
            sample["tube_detections"] = fo.Detections(detections=tubes)

        samples.append(sample)
    return samples


def _default_dataset_name(predictions_dir: Path) -> str:
    """Derive a dataset name from ``metrics.json``'s ``model_name`` field."""
    metrics = json.loads((predictions_dir / "metrics.json").read_text())
    model_name = metrics["model_name"]
    return f"{DATASET_NAME_PREFIX}-{model_name}-errors"


def build_dataset(
    predictions_dir: Path,
    export_dir: Path,
    dataset_name: str,
) -> fo.Dataset:
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
        logger.info("Deleted existing dataset '%s'.", dataset_name)

    index = json.loads((predictions_dir / "index.json").read_text())
    error_entries = [
        e for e in index if _classify(e["label"], e["is_positive"]) is not None
    ]
    logger.info(
        "Processing %d error sequences (of %d total).",
        len(error_entries),
        len(index),
    )

    samples: list[fo.Sample] = []
    for entry in tqdm(error_entries, desc="Building samples", unit="seq"):
        category = _classify(entry["label"], entry["is_positive"])
        assert category is not None  # filter above guarantees this

        record_path = predictions_dir / entry["json_path"]
        record = json.loads(record_path.read_text())
        seq_dir = (
            export_dir
            / LABEL_FOLDER[record["label"]]
            / record["subcategory"]
            / record["sequence_id"]
        )
        samples.extend(
            _build_samples_for_sequence(
                record=record,
                seq_dir=seq_dir,
                category=category,
            )
        )

    dataset = fo.Dataset(name=dataset_name, persistent=True)
    dataset.add_samples(samples)

    # Color scheme: YOLO priors purple, tubes red. Ignored by models that
    # don't populate the fields.
    dataset.app_config.color_scheme = fo.ColorScheme(
        fields=[
            {"path": "yolo_prior_detections", "fieldColor": "#8C00C8"},
            {"path": "tube_detections", "fieldColor": "#DC0000"},
        ]
    )
    dataset.save()

    logger.info(
        "Created dataset '%s' with %d samples from %d error sequences.",
        dataset_name,
        len(samples),
        len(error_entries),
    )
    return dataset


def main() -> None:
    args = _parse_args()
    dataset_name = args.dataset_name or _default_dataset_name(args.predictions_dir)
    build_dataset(
        predictions_dir=args.predictions_dir,
        export_dir=args.export_dir,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()

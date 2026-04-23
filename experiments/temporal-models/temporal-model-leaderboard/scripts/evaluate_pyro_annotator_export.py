"""Evaluate any registered TemporalModel on a pyro-annotator export.

Unlike ``scripts/evaluate.py`` (which runs on the flat ``sequential_test``
layout), this script targets the 3-level pyro-annotator export layout::

    <dataset-dir>/smoke/<subcategory>/<seq>/images/*.jpg          -> label="smoke"
    <dataset-dir>/false_positive/<subcategory>/<seq>/images/*.jpg -> label="fp"
    <dataset-dir>/unsure/<seq>/                                   -> skipped

The model is chosen via ``--model-type`` using the shared
:mod:`temporal_model_leaderboard.registry`, so any model wired into
the leaderboard works here.

Outputs mirror the ``evaluate.py`` convention (``index.json``,
``metrics.json``, ``dropped.json``) plus one per-sequence JSON under
``<output-dir>/<label>/<subcategory>/<seq>.json`` with ``trigger_frame_index``,
``ttd_frames``, and a pass-through of ``TemporalModelOutput.details``.
bbox-tube-specific fields (``kept_tubes``, ``tube_logits``,
``trigger_tube_id``) are hoisted to the top level so
``build_fiftyone_errors.py`` can render tube overlays without
special-casing the model.

Usage::

    uv run --group explore python scripts/evaluate_pyro_annotator_export.py \\
        --model-type bbox-tube-temporal \\
        --model-package data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip \\
        --export-dir data/01_raw/pyro_annotator_exports/sdis-77-new_model/sdis-77 \\
        --output-dir data/08_reporting/pyro_annotator/vit-dinov2-finetune \\
        --model-name bbox-tube-temporal-vit-dinov2-finetune
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from tqdm import tqdm

from temporal_model_leaderboard.dataset import get_sorted_frames
from temporal_model_leaderboard.metrics import compute_metrics
from temporal_model_leaderboard.registry import MODEL_REGISTRY, load_model
from temporal_model_leaderboard.types import SequenceResult

SMOKE_LABEL = "smoke"
FP_LABEL = "fp"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help="Registry key identifying which TemporalModel to load.",
    )
    parser.add_argument(
        "--model-package",
        type=Path,
        required=True,
        help="Path to the packaged model .zip file.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help=(
            "Root of the pyro-annotator export containing "
            "smoke/, false_positive/, and optional unsure/ sub-trees."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-sequence JSONs + index/metrics/dropped files.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Human-readable identifier written into metrics.json.",
    )
    return parser.parse_args()


def _discover(
    export_dir: Path,
) -> tuple[list[tuple[Path, str, str]], list[dict]]:
    """Return ``(kept, dropped)``.

    ``kept`` is a list of ``(seq_dir, label, subcategory)``; ``dropped`` lists
    sequences skipped up-front (``unsure/``).
    """
    kept: list[tuple[Path, str, str]] = []
    dropped: list[dict] = []

    smoke_root = export_dir / "smoke"
    if smoke_root.is_dir():
        for subcat_dir in sorted(p for p in smoke_root.iterdir() if p.is_dir()):
            for seq_dir in sorted(p for p in subcat_dir.iterdir() if p.is_dir()):
                kept.append((seq_dir, SMOKE_LABEL, subcat_dir.name))

    fp_root = export_dir / "false_positive"
    if fp_root.is_dir():
        for subcat_dir in sorted(p for p in fp_root.iterdir() if p.is_dir()):
            for seq_dir in sorted(p for p in subcat_dir.iterdir() if p.is_dir()):
                kept.append((seq_dir, FP_LABEL, subcat_dir.name))

    unsure_root = export_dir / "unsure"
    if unsure_root.is_dir():
        for seq_dir in sorted(p for p in unsure_root.iterdir() if p.is_dir()):
            dropped.append(
                {
                    "sequence_id": seq_dir.name,
                    "reason": "unsure",
                    "subcategory": None,
                }
            )

    return kept, dropped


def _bbox_tube_score(details: dict) -> float | None:
    """Return ``max(tube_logits)`` if present, else ``None``.

    bbox-tube's ``TemporalModelOutput.details`` carries the per-tube logits
    under ``details.tubes.kept[*].logit``. Other models don't populate this
    and we leave ``score`` as ``None``.
    """
    kept = details.get("tubes", {}).get("kept", [])
    if not kept:
        return None
    return max(float(t["logit"]) for t in kept)


def _record_to_json(
    *,
    sequence_id: str,
    label: str,
    subcategory: str,
    dataset_dir_rel: str,
    is_positive: bool,
    trigger_frame_index: int | None,
    ttd_frames: int | None,
    num_frames: int,
    details: dict,
) -> dict:
    """Shape the per-sequence JSON consumed by ``build_fiftyone_errors.py``.

    For bbox-tube models, the tube-specific fields inside ``details.tubes`` /
    ``details.decision`` are hoisted to the top level for convenient access
    by the FiftyOne renderer. Other models get ``kept_tubes: []`` etc.
    """
    tubes = details.get("tubes", {})
    decision = details.get("decision", {})
    kept = tubes.get("kept", [])
    return {
        "sequence_id": sequence_id,
        "label": label,
        "subcategory": subcategory,
        "dataset_dir_rel": dataset_dir_rel,
        "is_positive": is_positive,
        "trigger_frame_index": trigger_frame_index,
        "ttd_frames": ttd_frames,
        "num_frames": num_frames,
        "score": _bbox_tube_score(details),
        "num_tubes_kept": len(kept),
        "num_tubes_total": int(tubes.get("num_candidates", 0)),
        "tube_logits": [float(t["logit"]) for t in kept],
        "trigger_tube_id": decision.get("trigger_tube_id"),
        "threshold": (
            float(decision["threshold"]) if "threshold" in decision else None
        ),
        "kept_tubes": kept,
        "details": details,
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s from %s", args.model_type, args.model_package)
    model = load_model(args.model_type, args.model_package)

    logger.info("Discovering sequences under %s", args.export_dir)
    kept_seqs, dropped = _discover(args.export_dir)
    logger.info("Found %d sequences (+ %d dropped)", len(kept_seqs), len(dropped))

    results: list[SequenceResult] = []
    index_entries: list[dict] = []

    for seq_dir, label, subcategory in tqdm(
        kept_seqs, desc=args.model_name, unit="seq"
    ):
        frame_paths = get_sorted_frames(seq_dir)
        if not frame_paths:
            dropped.append(
                {
                    "sequence_id": seq_dir.name,
                    "reason": "no_images",
                    "subcategory": subcategory,
                }
            )
            continue

        frames = model.load_sequence(frame_paths)
        output = model.predict(frames)

        ground_truth = label == SMOKE_LABEL
        ttd_frames = (
            output.trigger_frame_index
            if ground_truth
            and output.is_positive
            and output.trigger_frame_index is not None
            else None
        )

        results.append(
            SequenceResult(
                sequence_id=seq_dir.name,
                ground_truth=ground_truth,
                predicted=output.is_positive,
                ttd_frames=ttd_frames,
            )
        )

        rec_json = _record_to_json(
            sequence_id=seq_dir.name,
            label=label,
            subcategory=subcategory,
            dataset_dir_rel=str(seq_dir.relative_to(args.export_dir)),
            is_positive=output.is_positive,
            trigger_frame_index=output.trigger_frame_index,
            ttd_frames=ttd_frames,
            num_frames=len(frames),
            details=output.details,
        )

        out_label_dir = args.output_dir / label / subcategory
        out_label_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_label_dir / f"{seq_dir.name}.json"
        out_path.write_text(json.dumps(rec_json, indent=2))

        index_entries.append(
            {
                "sequence_id": seq_dir.name,
                "label": label,
                "subcategory": subcategory,
                "is_positive": output.is_positive,
                "trigger_frame_index": output.trigger_frame_index,
                "ttd_frames": ttd_frames,
                "score": rec_json["score"],
                "json_path": str(out_path.relative_to(args.output_dir)),
            }
        )

    metrics = compute_metrics(args.model_name, results)

    # Sort index by score descending (None last).
    index_entries.sort(
        key=lambda e: (
            e["score"] is None,
            -(e["score"] if e["score"] is not None else 0.0),
        )
    )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(dataclasses.asdict(metrics), indent=2)
    )
    (args.output_dir / "index.json").write_text(json.dumps(index_entries, indent=2))
    (args.output_dir / "dropped.json").write_text(json.dumps(dropped, indent=2))

    logger.info("=== %s ===", args.model_name)
    logger.info(
        "  kept=%d dropped=%d  P=%.4f R=%.4f F1=%.4f FPR=%.4f",
        len(results),
        len(dropped),
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.fpr,
    )
    if metrics.mean_ttd_frames is not None:
        logger.info(
            "  Mean TTD=%.1f frames  Median TTD=%.1f frames",
            metrics.mean_ttd_frames,
            metrics.median_ttd_frames,
        )


if __name__ == "__main__":
    main()

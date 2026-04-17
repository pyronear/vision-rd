"""Run the full YOLO + tracking + classifier pipeline at package time.

Used by ``scripts/package_model.py`` to produce the training data for
``logistic_calibrator_fit.fit`` and the val data for probability-threshold
calibration. Bypasses the ``.zip`` entirely — we already have the YOLO
model, the classifier, and the config in memory at packaging time.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .data import get_sorted_frames, is_wf_sequence, list_sequences


def _iter_labelled_sequences(
    raw_dir: Path,
    model: Any,
) -> Iterator[tuple[str, str, list]]:
    """Yield ``(label, sequence_name, frames)`` for every sequence under
    ``raw_dir`` using the model's own ``load_sequence`` for frame decoding.

    Defined as a module-level helper so tests can monkey-patch it.
    """
    for seq_dir in list_sequences(raw_dir):
        frame_paths = get_sorted_frames(seq_dir)
        if not frame_paths:
            continue
        label = "smoke" if is_wf_sequence(seq_dir) else "fp"
        frames = model.load_sequence(frame_paths)
        yield label, seq_dir.name, frames


def collect_pipeline_records(
    *,
    model: Any,
    raw_dir: Path,
) -> list[dict]:
    """Run ``model.predict`` on every labelled sequence under ``raw_dir``.

    Returns a list of ``{"label", "sequence", "kept_tubes"}`` dicts with
    the same per-tube schema (``logit``, ``start_frame``, ``end_frame``,
    ``entries``) that ``scripts/analyze_variant.py`` and
    :mod:`bbox_tube_temporal.logistic_calibrator` expect.

    Args:
        model: A ``BboxTubeTemporalModel`` (or duck-type compatible) with
            ``.load_sequence(frame_paths) -> list[Frame]`` and
            ``.predict(frames) -> TemporalModelOutput`` whose
            ``.details["tubes"]["kept"]`` carries the tube structure.
        raw_dir: ``data/01_raw/datasets/{train,val}/`` with
            ``{wildfire,fp}/<seq>/images/*.jpg`` sub-trees.
    """
    records: list[dict] = []
    for label, seq_name, frames in _iter_labelled_sequences(raw_dir, model):
        out = model.predict(frames)
        kept = out.details.get("tubes", {}).get("kept", [])
        records.append(
            {
                "label": label,
                "sequence": seq_name,
                "kept_tubes": kept,
            }
        )
    return records

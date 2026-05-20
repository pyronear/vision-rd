"""Run configured temporal models over the sequence store and write results."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from bbox_tube_temporal.model import BboxTubeTemporalModel

from .outcomes import compute_outcome, decision_from_output, max_probability
from .store import build_frames, iter_sequence_dirs, read_meta

log = logging.getLogger(__name__)


def load_models(models_dir: Path, device: str = "cpu") -> dict:
    """Load every ``<name>/model.zip`` under the DVC-tracked ``models_dir``.

    Names are the subdirectory names; a subdir without a ``model.zip`` is skipped.
    """
    models: dict[str, object] = {}
    if not models_dir.exists():
        return models
    for sub in sorted(models_dir.iterdir()):
        pkg = sub / "model.zip"
        if not pkg.exists():
            log.warning("no model.zip under %s; skipping", sub)
            continue
        models[sub.name] = BboxTubeTemporalModel.from_package(pkg, device=device)
    return models


def run_over_store(
    store_dir: Path, models: dict, results_path: Path, details_dir: Path
) -> pd.DataFrame:
    """Run every model over every stored sequence; write results.parquet + details."""
    rows: list[dict] = []
    for seq_dir in iter_sequence_dirs(store_dir):
        meta = read_meta(seq_dir)
        frames = build_frames(seq_dir, meta)
        for name, model in models.items():
            t0 = time.perf_counter()
            out = model.predict(frames)
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            decision = decision_from_output(out.is_positive)
            tfile = None
            if (
                out.trigger_frame_index is not None
                and 0 <= out.trigger_frame_index < len(meta.frames)
            ):
                tfile = meta.frames[out.trigger_frame_index].file
            rows.append(
                {
                    "key": meta.key,
                    "source": meta.source,
                    "sequence_id": meta.sequence_id,
                    "camera_id": meta.camera_id,
                    "camera_name": meta.camera_name,
                    "organization_id": meta.organization_id,
                    "organization_name": meta.organization_name,
                    "label": meta.label,
                    "label_detail": meta.label_detail,
                    "n_frames": len(frames),
                    "model": name,
                    "decision": decision,
                    "trigger_frame_index": out.trigger_frame_index,
                    "trigger_frame_file": tfile,
                    "probability": max_probability(out.details),
                    "outcome": compute_outcome(decision, meta.label),
                    "runtime_ms": runtime_ms,
                }
            )
            model_details = details_dir / name
            model_details.mkdir(parents=True, exist_ok=True)
            (model_details / f"{meta.key}.json").write_text(
                json.dumps(out.details, indent=2, default=str)
            )
    df = pd.DataFrame(rows)
    # Use nullable StringDtype so None round-trips through to_dict("records") as None.
    for col in (
        "trigger_frame_file",
        "camera_name",
        "organization_name",
        "label_detail",
    ):
        if col in df.columns:
            df[col] = df[col].astype(pd.StringDtype())
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results_path)
    return df

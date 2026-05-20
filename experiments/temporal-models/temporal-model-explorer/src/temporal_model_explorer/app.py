"""Streamlit viewer over the explorer results (frontend-agnostic data layer).

Reads only data/07_model_output/{results.parquet,details/} +
data/03_primary/sequences/**; never runs models or fetches. Run with
`streamlit run app.py` (or `make app`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Absolute imports: this module is the Streamlit entrypoint, run as __main__ via
# `streamlit run app.py` (no package context), so relative imports would fail.
from temporal_model_explorer.outcomes import filter_results
from temporal_model_explorer.store import iter_sequence_dirs, read_meta

RESULTS = Path("data/07_model_output/results.parquet")
DETAILS = Path("data/07_model_output/details")
STORE = Path("data/03_primary/sequences")


def pivot_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sequence, one column of decision per model (+ carried meta)."""
    meta_cols = [
        "key",
        "source",
        "label",
        "label_detail",
        "camera_name",
        "organization_name",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    wide = df.pivot_table(
        index="key", columns="model", values="decision", aggfunc="first"
    ).reset_index()
    meta = df[meta_cols].drop_duplicates("key")
    return wide.merge(meta, on="key", how="left")


def decision_agreement(df: pd.DataFrame) -> pd.Series:
    """Per-key bool Series: do all models agree on the keep/discard decision?"""
    return df.groupby("key")["decision"].nunique() == 1


def load_details(model: str, key: str) -> dict:
    """Read the per-(model, sequence) details JSON; ``{}`` if absent."""
    path = DETAILS / model / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _find_seq_dir(key: str) -> Path | None:
    """Resolve a sequence by its meta `key` (folder names don't equal the key)."""
    for seq_dir in iter_sequence_dirs(STORE):
        if read_meta(seq_dir).key == key:
            return seq_dir
    return None


def main() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st  # noqa: PLC0415

    st.set_page_config(page_title="Temporal Model Explorer", layout="wide")
    st.title("Temporal Model Explorer — keep vs discard")

    if not RESULTS.exists():
        st.warning(
            "No results yet. Run `uv run dvc repro run_models` (or the run_models CLI)."
        )
        return
    df = pd.read_parquet(RESULTS)

    st.sidebar.header("Filters")

    def _opt(col):
        return [None, *sorted(x for x in df[col].dropna().unique())]

    model = st.sidebar.selectbox("model", _opt("model"))
    decision = st.sidebar.selectbox("decision", [None, "keep", "discard"])
    label = st.sidebar.selectbox("label", [None, "smoke", "fp", "unknown"])
    outcome = st.sidebar.selectbox("outcome", _opt("outcome"))
    source = st.sidebar.selectbox("source", _opt("source"))
    camera = st.sidebar.selectbox("camera", _opt("camera_name"))
    org = st.sidebar.selectbox("organization", _opt("organization_name"))
    errors_only = st.sidebar.checkbox("errors only (smoke discarded / fp kept)")

    view = filter_results(
        df,
        model=model,
        decision=decision,
        label=label,
        outcome=outcome,
        source=source,
        camera_name=camera,
        organization_name=org,
        errors_only=errors_only,
    )
    view_keys = set(view["key"].unique())

    # Cross-model agreement filter (only meaningful with >1 model).
    if df["model"].nunique() > 1:
        agreement = st.sidebar.selectbox("model agreement", [None, "agree", "disagree"])
        if agreement is not None:
            agree = decision_agreement(df)
            view_keys &= set(agree[agree == (agreement == "agree")].index)

    # Main table: one row per sequence, a decision column per model. All models
    # are shown for the selected sequences (even when filtering by one model), so
    # the table is a model-vs-model comparison.
    table = pivot_decisions(df[df["key"].isin(view_keys)])
    st.subheader(f"{len(view_keys)} sequences")
    st.dataframe(table, use_container_width=True)

    if not view_keys:
        return
    key = st.selectbox("Inspect a sequence", sorted(view_keys))
    seq_dir = _find_seq_dir(key)
    if seq_dir is None:
        return
    meta = read_meta(seq_dir)
    st.write(
        {
            "label": meta.label,
            "label_detail": meta.label_detail,
            "camera": meta.camera_name,
            "organization": meta.organization_name,
            "source": meta.source,
            "started_at": meta.started_at,
        }
    )
    seq_rows = df[df["key"] == key]
    st.dataframe(
        seq_rows[
            ["model", "decision", "outcome", "trigger_frame_index", "probability"]
        ],
        use_container_width=True,
    )
    for _, row in seq_rows.iterrows():
        with st.expander(f"details — {row['model']}"):
            st.json(load_details(row["model"], key))

    st.caption(
        "Frames in capture order. Per-frame bbox overlay + trigger highlight are "
        "deferred (need padded-index → input-frame mapping); see the spec Future work."
    )
    imgs = [str(seq_dir / f.file) for f in meta.frames]
    st.image(
        imgs, width=180, caption=[f"{i}: {Path(p).name}" for i, p in enumerate(imgs)]
    )


if __name__ == "__main__":  # pragma: no cover
    main()

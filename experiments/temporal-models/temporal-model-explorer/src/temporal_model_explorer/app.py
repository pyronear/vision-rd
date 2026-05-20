"""Streamlit viewer over the explorer results (frontend-agnostic data layer).

Reads only data/07_model_output/results.parquet + data/03_primary/sequences/**;
never runs models or fetches. Run with: `streamlit run app.py` (or `make app`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .outcomes import filter_results
from .store import iter_sequence_dirs, read_meta

RESULTS = Path("data/07_model_output/results.parquet")
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
    st.subheader(f"{view['key'].nunique()} sequences")
    st.dataframe(view, use_container_width=True)

    keys = sorted(view["key"].unique())
    if keys:
        key = st.selectbox("Inspect a sequence", keys)
        seq_dir = _find_seq_dir(key)
        if seq_dir:
            meta = read_meta(seq_dir)
            st.write(
                {
                    "label": meta.label,
                    "label_detail": meta.label_detail,
                    "camera": meta.camera_name,
                    "started_at": meta.started_at,
                }
            )
            st.dataframe(
                view[view["key"] == key][
                    ["model", "decision", "trigger_frame_index", "probability"]
                ]
            )
            imgs = [str(seq_dir / f.file) for f in meta.frames]
            st.image(imgs, width=180, caption=[Path(p).name for p in imgs])


if __name__ == "__main__":  # pragma: no cover
    main()

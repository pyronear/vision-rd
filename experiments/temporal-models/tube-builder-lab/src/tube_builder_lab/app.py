"""Tube Builder Lab — current vs candidate tube linking, Layout A.

Reads only data/05_model_input/detections/ + data/03_primary/sequences/**.
Run with `streamlit run app.py` (or `make app`). Iterate by editing
src/tube_builder_lab/candidate.py and clicking "Re-run candidate".
"""

from __future__ import annotations

import importlib
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

# Absolute imports: this module is the Streamlit entrypoint (run as __main__).
from tube_builder_lab import candidate as candidate_mod
from tube_builder_lab.cache import detections_present, load_cached
from tube_builder_lab.pipeline import (
    current_builder,
    detections_to_display_tubes,
    load_pipeline_config,
)
from tube_builder_lab.store import build_frames, read_meta, seq_dir_for_key
from tube_builder_lab.viz import (
    crop_tube_at_frame,
    draw_tube_bboxes,
    tube_color,
    tube_count,
    tube_timeline_df,
)
from tube_builder_lab.working_set import load_working_set

PARAMS = yaml.safe_load(Path("params.yaml").read_text())
STORE = Path(PARAMS["store"])
DETECTIONS = Path(PARAMS["detections_dir"])
PIPELINE_CONFIG = Path(PARAMS["pipeline_config"])
WORKING_SET = Path("working_set.yaml")
PLAY_FPS = 1


def _both_tube_sets(key: str, cfg, truncate: bool, _rev: int):
    """(current_tubes, candidate_tubes) for a key. _rev busts cache on Re-run."""
    fds = load_cached(DETECTIONS, key)
    cur = detections_to_display_tubes(fds, current_builder(cfg), cfg, truncate=truncate)
    cand = detections_to_display_tubes(
        fds, candidate_mod.build_tubes_candidate, cfg, truncate=truncate
    )
    return cur, cand


def _timeline_chart(tubes, n, current, color_map):  # pragma: no cover
    df = tube_timeline_df(tubes)
    order = sorted(df["tube"].unique(), key=lambda t: int(t[1:])) if len(df) else []
    xscale = alt.Scale(domain=[0, n], nice=False)
    bars = (
        alt.Chart(df)
        .mark_bar(height=16, cornerRadius=3)
        .encode(
            x=alt.X(
                "frame:Q",
                title="frame",
                scale=xscale,
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            x2="frame_end:Q",
            y=alt.Y("tube:N", title=None, sort=order),
            color=alt.Color(
                "tube:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[color_map[o] for o in order]),
                legend=None,
            ),
            opacity=alt.Opacity(
                "is_gap:N",
                scale=alt.Scale(domain=[False, True], range=[1.0, 0.4]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("tube:N"),
                alt.Tooltip("frame:Q"),
                alt.Tooltip("confidence:Q", format=".2f"),
                alt.Tooltip("is_gap:N"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [current + 0.5]}))
        .mark_rule(color="#111", strokeDash=[4, 3], size=2)
        .encode(x=alt.X("x:Q", scale=xscale, title="frame"))
    )
    return alt.layer(bars, rule).properties(
        height=max(70, len(order) * 30),
        autosize={"type": "fit-x", "contains": "padding"},
    )


@st.fragment(run_every=1.0 / PLAY_FPS)
def _viewer(key: str, cfg, truncate: bool, rev: int):  # pragma: no cover
    seq_dir = seq_dir_for_key(STORE, key)
    if seq_dir is None or not detections_present(DETECTIONS, key):
        st.warning(f"{key}: missing sequence frames or cached detections.")
        return
    meta = read_meta(seq_dir)
    frames = build_frames(seq_dir, meta)
    n = min(len(frames), cfg.max_frames) if truncate else len(frames)
    if not n:
        st.info("no frames")
        return

    cur, cand = _both_tube_sets(key, cfg, truncate, rev)
    c1, c2 = st.columns(2)
    c1.metric("current tubes", tube_count(cur))
    c2.metric(
        "candidate tubes", tube_count(cand), delta=tube_count(cand) - tube_count(cur)
    )

    frame_key = f"frame_{key}"
    st.session_state.setdefault(frame_key, 0)
    if st.toggle("▶ play", value=True, key=f"play_{key}"):
        st.session_state[frame_key] = (st.session_state[frame_key] + 1) % n
    i = st.slider("frame", 0, n - 1, key=frame_key) if n > 1 else 0

    # Two synced frame views (same frame index i): current (left) vs candidate
    # (right). One slider/play tick drives both, so they advance in lockstep.
    img_path = seq_dir / meta.frames[i].file
    left, right = st.columns(2)
    left.image(
        draw_tube_bboxes(img_path, cur, i),
        caption=f"current — frame {i + 1}/{n}, {tube_count(cur)} tube(s)",
        width="stretch",
    )
    right.image(
        draw_tube_bboxes(img_path, cand, i),
        caption=f"candidate — frame {i + 1}/{n}, {tube_count(cand)} tube(s)",
        width="stretch",
    )

    cur_colors = {f"T{t.tube_id}": tube_color(t.tube_id) for t in cur}
    cand_colors = {f"T{t.tube_id}": tube_color(t.tube_id) for t in cand}
    st.caption(f"current — {tube_count(cur)} tube(s)")
    if cur:
        st.altair_chart(_timeline_chart(cur, n, i, cur_colors), width="stretch")
    st.caption(f"candidate — {tube_count(cand)} tube(s)")
    if cand:
        st.altair_chart(_timeline_chart(cand, n, i, cand_colors), width="stretch")

    with st.expander("candidate crops @ this frame", expanded=False):
        cols = st.columns(max(1, len(cand)))
        for col, t in zip(cols, cand, strict=False):
            entry = next(
                (e for e in t.entries if e.frame_idx == i and e.detection), None
            )
            col.markdown(
                f"<b style='color:{tube_color(t.tube_id)}'>T{t.tube_id}</b>",
                unsafe_allow_html=True,
            )
            if entry:
                d = entry.detection
                col.image(
                    crop_tube_at_frame(
                        seq_dir / meta.frames[i].file, (d.cx, d.cy, d.w, d.h)
                    ),
                    width=180,
                )
            else:
                col.caption("inactive")


def _summary(cfg, truncate: bool, rev: int) -> pd.DataFrame:  # pragma: no cover
    rows = []
    ws = load_working_set(WORKING_SET)
    for item in ws.all():
        if not detections_present(DETECTIONS, item.key):
            rows.append(
                {
                    "key": item.key,
                    "current": None,
                    "candidate": None,
                    "Δ": None,
                    "note": item.note or "",
                }
            )
            continue
        cur, cand = _both_tube_sets(item.key, cfg, truncate, rev)
        rows.append(
            {
                "key": item.key,
                "current": len(cur),
                "candidate": len(cand),
                "Δ": len(cand) - len(cur),
                "note": item.note or "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:  # pragma: no cover
    st.set_page_config(page_title="Tube Builder Lab", layout="wide")
    st.title("Tube Builder Lab")

    if not PIPELINE_CONFIG.exists():
        st.warning("No pipeline config. Run `make cache` (cache_detections) first.")
        return
    cfg = load_pipeline_config(PIPELINE_CONFIG)
    ws = load_working_set(WORKING_SET)
    keys = [i.key for i in ws.all()]
    notes = {i.key: i.note for i in ws.all()}

    st.session_state.setdefault("rev", 0)
    with st.sidebar:
        st.header("Tube Lab")
        truncate = st.toggle(
            "truncate to max_frames",
            value=True,
            help=f"first {cfg.max_frames} frames (reproduces the model)",
        )
        if st.button("🔄 Re-run candidate"):
            importlib.reload(candidate_mod)
            st.session_state["rev"] += 1
            st.toast("candidate.py reloaded")
        idx = st.selectbox(
            "sequence",
            range(len(keys)),
            format_func=lambda j: (
                f"{keys[j]}  {('· ' + notes[keys[j]]) if notes[keys[j]] else ''}"
            ),
            key="seq_idx",
        )

    rev = st.session_state["rev"]
    key = keys[idx]
    if notes.get(key):
        st.caption(f"📝 {notes[key]}")
    _viewer(key, cfg, truncate, rev)

    st.divider()
    st.subheader("Working-set summary (current → candidate tube counts)")
    st.dataframe(_summary(cfg, truncate, rev), width="stretch", hide_index=True)


if __name__ == "__main__":  # pragma: no cover
    main()

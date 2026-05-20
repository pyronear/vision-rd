"""Streamlit viewer over the explorer results (frontend-agnostic data layer).

Reads only data/07_model_output/{results.parquet,details/} +
data/03_primary/sequences/**; never runs models or fetches. Run with
`streamlit run app.py` (or `make app`).

Left pane selects organization → camera → model. The main pane lists the
selected sequences (day-sorted); clicking a row opens it. The drill-down shows an
autoplaying (pausable) frame viewer with the YOLO bboxes overlaid and, alongside,
each extracted smoke tube as a context-cropped clip that autoplays on the same
playback tick, plus the temporal-model decision.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from bbox_tube_temporal.model_input import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
from PIL import Image, ImageDraw, ImageFont

# Absolute imports: this module is the Streamlit entrypoint, run as __main__ via
# `streamlit run app.py` (no package context), so relative imports would fail.
from temporal_model_explorer.store import iter_sequence_dirs, read_meta

RESULTS = Path("data/07_model_output/results.parquet")
DETAILS = Path("data/07_model_output/details")
STORE = Path("data/03_primary/sequences")
PARAMS = Path("params.yaml")

CROP_CONTEXT = 2.0  # bbox expansion for tube crops (more context than the model's 1.5)
CROP_SIZE = 224
PLAY_FPS = 1  # autoplay speed (frames/sec); fixed, no UI control

try:
    _BBOX_FONT = ImageFont.load_default(size=18)  # confidence labels on the frame
except TypeError:  # older Pillow without the size kwarg
    _BBOX_FONT = ImageFont.load_default()

# Display vocabulary. The underlying columns stay label/decision/outcome; the UI
# shows: ground truth (label) · model verdict (decision) · correctness (outcome).
CORRECTNESS = {
    "kept-smoke": "✅ smoke kept",
    "discarded-fp": "✅ fp filtered",
    "discarded-smoke": "🔴 missed smoke",
    "kept-fp": "🟠 false alarm",
    "n/a": "—",
}
ROW_BG = {  # by correctness (errors stand out)
    "🔴 missed smoke": "#f4b4b4",
    "🟠 false alarm": "#fbdca0",
    "✅ smoke kept": "#bfe7bf",
    "✅ fp filtered": "#e6f2e6",
}
KEEP_BG = "#cfe2ff"  # flagged as smoke, ground truth unknown
DISCARD_BG = "#eeeeee"  # discarded, ground truth unknown


def registered_models() -> list[str]:
    """Model names from params.yaml (so the UI only offers configured models)."""
    if not PARAMS.exists():
        return []
    return list((yaml.safe_load(PARAMS.read_text()) or {}).get("models", {}).keys())


def day_of(started_at: str | None) -> str:
    """Calendar day (YYYY-MM-DD) from an ISO timestamp; 'unknown' if absent."""
    return started_at[:10] if started_at else "unknown"


def correctness_label(outcome: str) -> str:
    """Human-friendly correctness label for a raw outcome value."""
    return CORRECTNESS.get(outcome, outcome)


def row_background(verdict: str, correctness: str) -> str:
    """Row background colour from the model verdict + correctness label.

    Errors stand out (missed smoke / false alarm); correct rows are green; rows
    with unknown ground truth are tinted by the verdict (kept vs discarded).
    """
    return ROW_BG.get(correctness) or (KEEP_BG if verdict == "keep" else DISCARD_BG)


TUBE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def tube_color(tube_id: int) -> str:
    """Stable colour for a tube id (same colour in the timeline + crop headers)."""
    return TUBE_PALETTE[tube_id % len(TUBE_PALETTE)]


def legend_html() -> str:
    """HTML chips explaining the table row colours (built from the colour map)."""
    items = [
        ("🔴 missed smoke (real smoke discarded)", ROW_BG["🔴 missed smoke"]),
        ("🟠 false alarm (fp kept)", ROW_BG["🟠 false alarm"]),
        ("✅ smoke kept", ROW_BG["✅ smoke kept"]),
        ("✅ fp filtered", ROW_BG["✅ fp filtered"]),
        ("flagged smoke · GT unknown", KEEP_BG),
        ("discarded · GT unknown", DISCARD_BG),
    ]
    chips = "".join(
        f'<span style="background:{color};color:#111;padding:2px 8px;'
        f'border-radius:4px;margin:0 6px 4px 0;display:inline-block">{label}</span>'
        for label, color in items
    )
    return f'<div style="line-height:2.2">{chips}</div>'


def processed_to_input_index(
    frame_idx: int, padded_frame_indices: list[int]
) -> int | None:
    """Map a model-processed frame index back to the input-frame index.

    The model truncates/pads the sequence; ``padded_frame_indices`` are the
    synthetic (duplicate) slots. A real slot's input index is its position minus
    the number of synthetic slots before it. Returns ``None`` for synthetic slots.
    """
    if frame_idx in padded_frame_indices:
        return None
    return frame_idx - sum(1 for p in padded_frame_indices if p < frame_idx)


def frame_bboxes_by_input_index(details: dict) -> dict[int, list[tuple]]:
    """input-frame index → list of ((cx,cy,w,h), confidence) from kept tubes."""
    padded = (details or {}).get("preprocessing", {}).get("padded_frame_indices", [])
    out: dict[int, list[tuple]] = {}
    for tube in (details or {}).get("tubes", {}).get("kept", []):
        for entry in tube.get("entries", []):
            if entry.get("bbox") is None:
                continue
            inp = processed_to_input_index(entry["frame_idx"], padded)
            if inp is None:
                continue
            out.setdefault(inp, []).append(
                (tuple(entry["bbox"]), entry.get("confidence"))
            )
    return out


def tube_input_boxes(
    tube: dict, padded_frame_indices: list[int]
) -> list[tuple[int, tuple]]:
    """(input_index, bbox) for one tube's real (non-synthetic) detected entries."""
    boxes: list[tuple[int, tuple]] = []
    for entry in tube.get("entries", []):
        if entry.get("bbox") is None:
            continue
        inp = processed_to_input_index(entry["frame_idx"], padded_frame_indices)
        if inp is not None:
            boxes.append((inp, tuple(entry["bbox"])))
    return boxes


def draw_bboxes(
    image_path: Path, boxes, color: str = "red", width: int = 4
) -> Image.Image:
    """Return the frame with bboxes drawn; ``boxes`` is ``[((cx,cy,w,h), conf), ...]``.

    The confidence (when present) is printed just above each box.
    """
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)
    for (cx, cy, bw, bh), conf in boxes:
        x0, y0 = (cx - bw / 2) * w_img, (cy - bh / 2) * h_img
        x1, y1 = (cx + bw / 2) * w_img, (cy + bh / 2) * h_img
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        if conf is not None:
            draw.text((x0, max(0, y0 - 20)), f"{conf:.2f}", fill=color, font=_BBOX_FONT)
    return img


def crop_around_bbox(
    image_path: Path,
    bbox_norm,
    context_factor: float = CROP_CONTEXT,
    patch_size: int = CROP_SIZE,
) -> Image.Image:
    """Square crop centred on a normalized bbox, expanded for context (reuses the
    lib's model-input crop so it matches what the classifier sees)."""
    img = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w = img.shape[:2]
    cx, cy, bw, bh = bbox_norm
    ecx, ecy, ew, eh = expand_bbox(cx, cy, bw, bh, context_factor)
    box = norm_bbox_to_pixel_square(ecx, ecy, ew, eh, img_w, img_h)
    return Image.fromarray(crop_and_resize(img, box, patch_size))


def tube_timeline_df(tube_rows: list[tuple[str, set]]) -> pd.DataFrame:
    """Long frame for the Altair tube timeline: one row per (tube, present frame).

    ``tube_rows`` is ``[(label, frame_index_set), ...]``; ``frame_end`` = frame + 1
    so each present frame renders as a unit-width bar.
    """
    records = [
        {"tube": label, "frame": f, "frame_end": f + 1}
        for label, frames in tube_rows
        for f in sorted(frames)
    ]
    return pd.DataFrame(records, columns=["tube", "frame", "frame_end"])


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


def started_at_by_key() -> dict[str, str | None]:
    """key → started_at from the store metas (used when results lack the column)."""
    out: dict[str, str | None] = {}
    for seq_dir in iter_sequence_dirs(STORE):
        meta = read_meta(seq_dir)
        out[meta.key] = meta.started_at
    return out


def _tube_timeline_chart(
    alt, tube_rows, n, trigger, current, color_map, trigger_tube_id=None
):  # pragma: no cover
    """One colour-coded bar row per tube + trigger/current rules (Altair).

    The trigger tube's bars get a dark outline so it stands out.
    """
    order = [label for label, _ in tube_rows]
    xscale = alt.Scale(domain=[0, n], nice=False)
    bars = (
        alt.Chart(tube_timeline_df(tube_rows))
        .mark_bar(height=16, cornerRadius=3)
        .encode(
            x=alt.X(
                "frame:Q",
                title="frame",
                scale=xscale,
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            x2="frame_end:Q",
            y=alt.Y("tube:N", title=None, sort=order, axis=alt.Axis(labelFontSize=13)),
            color=alt.Color(
                "tube:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[color_map[o] for o in order]),
                legend=None,
            ),
            tooltip=["tube", "frame"],
        )
    )
    if trigger_tube_id is not None:
        is_trig = f"datum.tube === 'T{trigger_tube_id}'"
        bars = bars.encode(
            stroke=alt.condition(is_trig, alt.value("#111"), alt.value(None)),
            strokeWidth=alt.condition(is_trig, alt.value(2.5), alt.value(0)),
        )
    layers = [bars]
    if trigger is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"x": [trigger + 0.5]}))
            .mark_rule(color="#c62828", size=2)
            .encode(x=alt.X("x:Q", scale=xscale, axis=None))
        )
    layers.append(
        alt.Chart(pd.DataFrame({"x": [current + 0.5]}))
        .mark_rule(color="#111", strokeDash=[4, 3], size=2)
        .encode(x=alt.X("x:Q", scale=xscale, title="frame"))
    )
    return alt.layer(*layers).properties(height=max(90, len(tube_rows) * 34))


def _drilldown(st, key: str, model: str, row: pd.Series) -> None:  # pragma: no cover
    import altair as alt  # noqa: PLC0415

    seq_dir = _find_seq_dir(key)
    if seq_dir is None:
        st.warning(f"sequence {key} not found in the store")
        return
    meta = read_meta(seq_dir)
    details = load_details(model, key)
    bbmap = frame_bboxes_by_input_index(details)
    padded = details.get("preprocessing", {}).get("padded_frame_indices", [])
    kept = details.get("tubes", {}).get("kept", [])
    n = len(meta.frames)
    trig_raw = row["trigger_frame_index"]
    trig = (
        processed_to_input_index(int(trig_raw), padded) if pd.notna(trig_raw) else None
    )

    is_keep = row["decision"] == "keep"
    verdict = "💨 KEEP (smoke)" if is_keep else "🚫 DISCARD (no smoke)"
    st.subheader(f"{verdict} — {key}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("model verdict", row["decision"])
    c2.metric("correctness", correctness_label(row["outcome"]))
    c3.metric("trigger frame", "—" if trig is None else str(trig))
    prob = row["probability"]
    c4.metric("probability", f"{prob:.3f}" if pd.notna(prob) else "—")
    st.caption(
        f"ground truth={meta.label} ({meta.label_detail}) · "
        f"camera={meta.camera_name} · org={meta.organization_name} · frames={n}"
    )
    if not n:
        return

    # Playback: a compact play/pause toggle (its own short line) above a full-width
    # timeline + slider. While playing we advance the slider's session_state value
    # BEFORE the slider widget is instantiated (the only point at which modifying a
    # widget-keyed value is allowed).
    frame_key = f"frame_{key}"
    st.session_state.setdefault(frame_key, 0)
    playing = st.toggle("▶ play", value=True, key=f"play_{key}")
    if playing:
        st.session_state[frame_key] = (st.session_state[frame_key] + 1) % n
    cur = st.session_state[frame_key] % n

    trigger_tube_id = details.get("decision", {}).get("trigger_tube_id")
    tube_rows = [
        (f"T{t['tube_id']}", {idx for idx, _ in tube_input_boxes(t, padded)})
        for t in kept
    ]
    color_map = {f"T{t['tube_id']}": tube_color(t["tube_id"]) for t in kept}
    if tube_rows:
        st.altair_chart(
            _tube_timeline_chart(
                alt, tube_rows, n, trig, cur, color_map, trigger_tube_id
            ),
            use_container_width=True,
        )
    else:
        st.info("no smoke tubes extracted")
    i = st.slider("frame", 0, n - 1, key=frame_key) if n > 1 else 0

    frame_col, tubes_col = st.columns([2, 1])
    ref = meta.frames[i]
    frame_col.image(
        draw_bboxes(seq_dir / ref.file, bbmap.get(i, [])),
        caption=f"frame {i + 1}/{n} — {len(bbmap.get(i, []))} detection(s)",
        use_container_width=True,
    )

    # Each tube crop is synced to the current frame i; the trigger tube is badged.
    # Render into a single st.empty() placeholder so a sequence with fewer tubes
    # fully replaces the previous one (no stale/ghosted leftovers across reruns).
    with tubes_col.empty().container():
        st.markdown(f"**tubes @ frame {i}** (context-cropped)")
        for tube in kept:
            at_frame = dict(tube_input_boxes(tube, padded))
            color = tube_color(tube["tube_id"])
            tprob = tube.get("probability")
            stat = (
                f"prob {tprob:.2f}"
                if tprob is not None
                else f"logit {tube['logit']:.2f}"
            )
            badge = " ⚡<b>triggered</b>" if tube["tube_id"] == trigger_tube_id else ""
            chip = f"<b style='color:{color}'>● T{tube['tube_id']}</b>"
            st.markdown(f"{chip} · {stat}{badge}", unsafe_allow_html=True)
            if i in at_frame:
                st.image(
                    crop_around_bbox(seq_dir / meta.frames[i].file, at_frame[i]),
                    width=220,
                )
            else:
                st.caption("inactive at this frame")

    if playing:
        time.sleep(1.0 / PLAY_FPS)
        st.rerun()


def main() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st  # noqa: PLC0415

    st.set_page_config(page_title="Temporal Model Explorer", layout="wide")
    st.title("Temporal Model Explorer")

    if not RESULTS.exists():
        st.warning(
            "No results yet. Run `uv run dvc repro run_models` (or the run_models CLI)."
        )
        return
    df = pd.read_parquet(RESULTS)
    if "started_at" not in df.columns:
        # Cache the store scan once per session so autoplay reruns stay snappy.
        if "started_at_map" not in st.session_state:
            st.session_state["started_at_map"] = started_at_by_key()
        df["started_at"] = df["key"].map(st.session_state["started_at_map"])
    models = [m for m in registered_models() if m in set(df["model"])] or sorted(
        df["model"].unique()
    )

    st.sidebar.header("Select")
    orgs = sorted(x for x in df["organization_name"].dropna().unique())
    org = st.sidebar.selectbox("organization", orgs, key="org") if orgs else None
    org_df = df[df["organization_name"] == org] if org else df
    cameras = sorted(x for x in org_df["camera_name"].dropna().unique())
    camera = st.sidebar.selectbox("camera", cameras, key="camera") if cameras else None
    model = st.sidebar.selectbox("model", models, key="model")

    view = df[df["model"] == model]
    if org:
        view = view[view["organization_name"] == org]
    if camera:
        view = view[view["camera_name"] == camera]
    view = (
        view.assign(day=view["started_at"].map(day_of))
        .sort_values("started_at", ascending=False)
        .reset_index(drop=True)
    )

    head, filt = st.columns([6, 1])
    with filt.popover("🔎 filter"):
        f_gt = st.selectbox(
            "ground truth", ["all", "smoke", "fp", "unknown"], key="f_gt"
        )
        f_mv = st.selectbox("model verdict", ["all", "keep", "discard"], key="f_mv")
        f_corr = st.selectbox(
            "correctness", ["all", *CORRECTNESS.values()], key="f_corr"
        )
    if f_gt != "all":
        view = view[view["label"] == f_gt]
    if f_mv != "all":
        view = view[view["decision"] == f_mv]
    if f_corr != "all":
        view = view[view["outcome"].map(correctness_label) == f_corr]

    head.subheader(f"{len(view)} sequences — {camera or 'all cameras'}")
    if view.empty:
        return

    display = view.assign(correctness=view["outcome"].map(correctness_label)).rename(
        columns={"label": "ground truth", "decision": "model verdict"}
    )
    cols = [
        "day",
        "key",
        "started_at",
        "ground truth",
        "model verdict",
        "correctness",
        "probability",
    ]

    def _style_row(r):
        bg = row_background(r["model verdict"], r["correctness"])
        return [f"background-color: {bg}; color: #111"] * len(cols)

    styled = display[cols].style.apply(_style_row, axis=1)
    st.markdown(legend_html(), unsafe_allow_html=True)
    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="seqtable",
    )
    # Persist the selection in session_state: autoplay's st.rerun() doesn't carry
    # the table's selection event, so without this the viewer would snap back to
    # the first row every second.
    rows = event.selection.rows
    if rows and rows[0] < len(view):
        st.session_state["selected_key"] = view.iloc[rows[0]]["key"]
    selected = st.session_state.get("selected_key")
    if selected not in set(view["key"]):
        selected = view.iloc[0]["key"]
    _drilldown(st, selected, model, view[view["key"] == selected].iloc[0])


if __name__ == "__main__":  # pragma: no cover
    main()

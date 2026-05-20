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
from PIL import Image, ImageDraw

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
    """input-frame index → list of normalized (cx,cy,w,h) bboxes from kept tubes."""
    padded = (details or {}).get("preprocessing", {}).get("padded_frame_indices", [])
    out: dict[int, list[tuple]] = {}
    for tube in (details or {}).get("tubes", {}).get("kept", []):
        for entry in tube.get("entries", []):
            if entry.get("bbox") is None:
                continue
            inp = processed_to_input_index(entry["frame_idx"], padded)
            if inp is None:
                continue
            out.setdefault(inp, []).append(tuple(entry["bbox"]))
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
    image_path: Path, bboxes_norm, color: str = "red", width: int = 4
) -> Image.Image:
    """Return the frame with normalized (cx,cy,w,h) bboxes drawn on it."""
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)
    for cx, cy, bw, bh in bboxes_norm:
        x0, y0 = (cx - bw / 2) * w_img, (cy - bh / 2) * h_img
        x1, y1 = (cx + bw / 2) * w_img, (cy + bh / 2) * h_img
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
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


def timeline_image(
    n_frames: int,
    active_frames,
    trigger: int | None = None,
    current: int | None = None,
    width: int = 1000,
    height: int = 18,
) -> Image.Image:
    """Thin timeline bar aligned above the slider: green where a smoke tube is
    present, red at the trigger frame, a dark tick at the current frame, grey
    elsewhere."""
    img = Image.new("RGB", (width, height), "#e8e8e8")
    if n_frames <= 0:
        return img
    draw = ImageDraw.Draw(img)
    active = set(active_frames)
    seg = width / n_frames
    for i in range(n_frames):
        if i in active:
            draw.rectangle(
                [int(i * seg), 0, max(int(i * seg), int((i + 1) * seg) - 1), height - 1],
                fill="#2e7d32",
            )
    if trigger is not None and 0 <= trigger < n_frames:
        draw.rectangle(
            [
                int(trigger * seg),
                0,
                max(int(trigger * seg), int((trigger + 1) * seg) - 1),
                height - 1,
            ],
            fill="#c62828",
        )
    if current is not None and 0 <= current < n_frames:
        x = int((current + 0.5) * seg)
        draw.rectangle([max(0, x - 1), 0, min(width - 1, x + 1), height - 1], fill="#000")
    return img


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


def _drilldown(st, key: str, model: str, row: pd.Series) -> None:  # pragma: no cover
    seq_dir = _find_seq_dir(key)
    if seq_dir is None:
        st.warning(f"sequence {key} not found in the store")
        return
    meta = read_meta(seq_dir)
    details = load_details(model, key)

    is_smoke = row["decision"] == "keep"
    st.subheader(
        f"{'🟥 KEEP (smoke)' if is_smoke else '⬜ DISCARD (no smoke)'} — {key}"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("model verdict", row["decision"])
    c2.metric("correctness", correctness_label(row["outcome"]))
    c3.metric("trigger frame idx", str(row["trigger_frame_index"]))
    prob = row["probability"]
    c4.metric("probability", f"{prob:.3f}" if pd.notna(prob) else "—")
    st.caption(
        f"ground truth={meta.label} ({meta.label_detail}) · "
        f"camera={meta.camera_name} · org={meta.organization_name} · "
        f"frames={len(meta.frames)}"
    )

    bbmap = frame_bboxes_by_input_index(details)
    padded = details.get("preprocessing", {}).get("padded_frame_indices", [])
    kept = details.get("tubes", {}).get("kept", [])
    n = len(meta.frames)
    if not n:
        return

    # Playback: a play/pause toggle + a frame slider. While playing we advance the
    # slider's session_state value BEFORE the slider widget is instantiated (the
    # only point at which modifying a widget-keyed value is allowed).
    frame_key = f"frame_{key}"
    st.session_state.setdefault(frame_key, 0)
    top = st.columns([1, 9])
    playing = top[0].toggle("▶ play", value=True, key=f"play_{key}")
    if playing:
        st.session_state[frame_key] = (st.session_state[frame_key] + 1) % n
    cur = st.session_state[frame_key] % n
    trig_raw = row["trigger_frame_index"]
    trig = (
        processed_to_input_index(int(trig_raw), padded) if pd.notna(trig_raw) else None
    )
    top[1].image(
        timeline_image(n, set(bbmap), trig, cur),
        use_container_width=True,
        caption="🟩 smoke tube present · 🟥 trigger · ▏ current frame",
    )
    i = top[1].slider("frame", 0, n - 1, key=frame_key) if n > 1 else 0

    frame_col, tubes_col = st.columns([2, 1])
    ref = meta.frames[i]
    frame_col.image(
        draw_bboxes(seq_dir / ref.file, bbmap.get(i, [])),
        caption=f"frame {i + 1}/{n} — {Path(ref.file).name} — "
        f"{len(bbmap.get(i, []))} detection(s)",
        use_container_width=True,
    )

    # Each tube crop is synced to the current frame i (shown when the tube has a
    # detection at that frame).
    tubes_col.markdown(f"**{len(kept)} smoke tube(s)** — context crop @ frame {i}")
    for tube in kept:
        at_frame = dict(tube_input_boxes(tube, padded))
        tprob = tube.get("probability")
        head = (
            f"tube {tube['tube_id']} · prob {tprob:.2f}"
            if tprob is not None
            else f"tube {tube['tube_id']} · logit {tube['logit']:.2f}"
        )
        if i in at_frame:
            tubes_col.image(
                crop_around_bbox(seq_dir / meta.frames[i].file, at_frame[i]),
                width=220,
                caption=head,
            )
        else:
            tubes_col.caption(f"{head} — inactive at frame {i}")

    with st.expander("raw details JSON"):
        st.json(details)

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

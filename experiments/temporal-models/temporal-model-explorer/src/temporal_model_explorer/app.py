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


def registered_models() -> list[str]:
    """Model names from params.yaml (so the UI only offers configured models)."""
    if not PARAMS.exists():
        return []
    return list((yaml.safe_load(PARAMS.read_text()) or {}).get("models", {}).keys())


def day_of(started_at: str | None) -> str:
    """Calendar day (YYYY-MM-DD) from an ISO timestamp; 'unknown' if absent."""
    return started_at[:10] if started_at else "unknown"


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
    c1.metric("decision", row["decision"])
    c2.metric("outcome", str(row["outcome"]))
    c3.metric("trigger frame idx", str(row["trigger_frame_index"]))
    prob = row["probability"]
    c4.metric("probability", f"{prob:.3f}" if pd.notna(prob) else "—")
    st.caption(
        f"label={meta.label} ({meta.label_detail}) · camera={meta.camera_name} · "
        f"org={meta.organization_name} · frames={len(meta.frames)}"
    )

    bbmap = frame_bboxes_by_input_index(details)
    padded = details.get("preprocessing", {}).get("padded_frame_indices", [])
    kept = details.get("tubes", {}).get("kept", [])
    n = len(meta.frames)
    if not n:
        return

    # One playback tick drives the full-frame view AND every tube crop.
    tick_key = f"tick_{key}"
    st.session_state.setdefault(tick_key, 0)
    ctrl = st.columns([1, 1, 1, 3])
    playing = ctrl[0].toggle("▶ play", value=True, key=f"play_{key}")
    if ctrl[1].button("⏮", key=f"prev_{key}"):
        st.session_state[tick_key] -= 1
    if ctrl[2].button("⏭", key=f"next_{key}"):
        st.session_state[tick_key] += 1
    fps = ctrl[3].slider("fps", 1, 10, 4, key=f"fps_{key}")
    tick = st.session_state[tick_key]

    frame_col, tubes_col = st.columns([2, 1])
    i = tick % n
    ref = meta.frames[i]
    frame_col.image(
        draw_bboxes(seq_dir / ref.file, bbmap.get(i, [])),
        caption=f"frame {i + 1}/{n} — {Path(ref.file).name} — "
        f"{len(bbmap.get(i, []))} detection(s)",
        use_container_width=True,
    )

    tubes_col.markdown(f"**{len(kept)} smoke tube(s)** (context-cropped)")
    for tube in kept:
        boxes = tube_input_boxes(tube, padded)
        if not boxes:
            continue
        ti = tick % len(boxes)
        in_idx, bbox = boxes[ti]
        tprob = tube.get("probability")
        head = (
            f"tube {tube['tube_id']} · prob {tprob:.2f}"
            if tprob is not None
            else f"tube {tube['tube_id']} · logit {tube['logit']:.2f}"
        )
        tubes_col.image(
            crop_around_bbox(seq_dir / meta.frames[in_idx].file, bbox),
            width=220,
            caption=f"{head} — f{in_idx} ({ti + 1}/{len(boxes)})",
        )

    with st.expander("raw details JSON"):
        st.json(details)

    if playing:
        st.session_state[tick_key] = tick + 1
        time.sleep(1.0 / fps)
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

    st.subheader(f"{len(view)} sequences — {camera or 'all cameras'}")
    if view.empty:
        return
    cols = ["day", "key", "started_at", "label", "decision", "outcome", "probability"]
    event = st.dataframe(
        view[cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="seqtable",
    )
    rows = event.selection.rows
    pos = rows[0] if rows and rows[0] < len(view) else 0
    _drilldown(st, view.iloc[pos]["key"], model, view.iloc[pos])


if __name__ == "__main__":  # pragma: no cover
    main()

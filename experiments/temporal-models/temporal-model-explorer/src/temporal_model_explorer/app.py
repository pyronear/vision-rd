"""Streamlit viewer over the explorer results (frontend-agnostic data layer).

Reads only data/07_model_output/{results.parquet,details/} +
data/03_primary/sequences/**; never runs models or fetches. Run with
`streamlit run app.py` (or `make app`).

Left pane selects organization → camera → model. The main pane lists the
selected sequences grouped by day; selecting one shows a frame slider with the
YOLO bboxes overlaid, the extracted smoke tubes, and the temporal-model decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageDraw

# Absolute imports: this module is the Streamlit entrypoint, run as __main__ via
# `streamlit run app.py` (no package context), so relative imports would fail.
from temporal_model_explorer.store import iter_sequence_dirs, read_meta

RESULTS = Path("data/07_model_output/results.parquet")
DETAILS = Path("data/07_model_output/details")
STORE = Path("data/03_primary/sequences")
PARAMS = Path("params.yaml")


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
        f"org={meta.organization_name} · frames={len(meta.frames)} · "
        f"kept tubes={len(details.get('tubes', {}).get('kept', []))}"
    )

    # Frame slider with YOLO bboxes overlaid.
    bbmap = frame_bboxes_by_input_index(details)
    n = len(meta.frames)
    if n:
        i = st.slider("frame", 0, n - 1, 0) if n > 1 else 0
        ref = meta.frames[i]
        st.image(
            draw_bboxes(seq_dir / ref.file, bbmap.get(i, [])),
            caption=f"frame {i}/{n - 1} — {Path(ref.file).name} — "
            f"{len(bbmap.get(i, []))} detection(s)",
            use_container_width=True,
        )

    # Extracted smoke tubes.
    padded = details.get("preprocessing", {}).get("padded_frame_indices", [])
    kept = details.get("tubes", {}).get("kept", [])
    st.markdown(f"### Extracted smoke tubes ({len(kept)})")
    for tube in kept:
        prob = tube.get("probability")
        st.markdown(
            f"**Tube {tube['tube_id']}** — frames {tube['start_frame']}–"
            f"{tube['end_frame']} · logit {tube['logit']:.2f} · "
            f"prob {prob:.3f} · first crossing {tube['first_crossing_frame']}"
            if prob is not None
            else f"**Tube {tube['tube_id']}** — frames {tube['start_frame']}–"
            f"{tube['end_frame']} · logit {tube['logit']:.2f} · "
            f"first crossing {tube['first_crossing_frame']}"
        )
        boxes = tube_input_boxes(tube, padded)
        imgs = [
            draw_bboxes(seq_dir / meta.frames[idx].file, [bbox]) for idx, bbox in boxes
        ]
        if imgs:
            st.image(imgs, width=130, caption=[f"f{idx}" for idx, _ in boxes])

    with st.expander("raw details JSON"):
        st.json(details)


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
        df["started_at"] = df["key"].map(started_at_by_key())
    models = [m for m in registered_models() if m in set(df["model"])] or sorted(
        df["model"].unique()
    )

    st.sidebar.header("Select")
    orgs = sorted(x for x in df["organization_name"].dropna().unique())
    org = st.sidebar.selectbox("organization", orgs) if orgs else None
    org_df = df[df["organization_name"] == org] if org else df
    cameras = sorted(x for x in org_df["camera_name"].dropna().unique())
    camera = st.sidebar.selectbox("camera", cameras) if cameras else None
    model = st.sidebar.selectbox("model", models)

    view = df[df["model"] == model]
    if org:
        view = view[view["organization_name"] == org]
    if camera:
        view = view[view["camera_name"] == camera]

    view = view.assign(day=view["started_at"].map(day_of))
    st.subheader(f"{len(view)} sequences — {camera or 'all cameras'}")
    cols = ["key", "started_at", "label", "decision", "outcome", "probability"]
    for day in sorted(view["day"].unique(), reverse=True):
        day_rows = view[view["day"] == day]
        st.markdown(f"**{day}** — {len(day_rows)} sequences")
        st.dataframe(day_rows[cols], use_container_width=True, hide_index=True)

    if view.empty:
        return
    key = st.selectbox("Inspect a sequence", sorted(view["key"]))
    _drilldown(st, key, model, view[view["key"] == key].iloc[0])


if __name__ == "__main__":  # pragma: no cover
    main()

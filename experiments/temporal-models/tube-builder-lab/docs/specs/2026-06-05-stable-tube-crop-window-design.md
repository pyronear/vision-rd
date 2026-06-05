# Per-tube stable crop window + film-strip viz

Date: 2026-06-05
Status: approved (brainstorm)

## Problem

The crop the temporal head sees is computed **per-frame**. In
`bbox_tube_temporal/model_input.py::process_tube`, each tube entry's own bbox is
expanded by `context_factor`, squared, and resized to 224. As a plume grows and
drifts across the 30s-apart frames, the crop **recenters and rescales every
frame**: the smoke stays roughly centered and same-size while the *background*
slides and zooms. The result is a "jumpy" sequence — the temporal head never
sees the smoke actually move.

We want the opposite: a **single fixed crop window per tube**, applied to every
frame, so the background stays static and the smoke visibly moves and grows
inside it.

## Key decisions

- **What is fixed:** position *and* size — one fixed window for the whole tube
  (not size-only, not a smoothed/EMA window). Strongest "stable image, moving
  smoke" effect.
- **How the window is derived:** the **union (enclosing) box** of all the tube's
  observed detections, expanded by a context margin. Guarantees the smoke is
  always inside the crop. Early frames show a small plume in a large window —
  acceptable and expected.
- **Placement (approach A → B):** stabilization is a **crop-window decision**,
  not a tube-*building* step. It does not change which detections link into
  which tube. Implement it as a **pure function that leaves `Tube` untouched**,
  used only at crop time. Mutating entry bboxes would corrupt the timeline,
  per-frame confidence, IoU/merge logic, and gap interpolation.
  - **A (this spec):** lab-only pure function + film-strip view; no changes to
    `bbox_tube_temporal`.
  - **B (later follow-up):** lift `tube_window` into `process_tube` as the real
    temporal-head crop. Out of scope here.
- **Viz:** a per-tube **film-strip comparison** — two thumbnail rows spanning
  the tube, top = current per-frame crop (jumpy), bottom = stabilized
  fixed-window crop.

### Note on the eventual production step (B)

When stable crops are eventually fed to the temporal head, that is a different
input distribution than the head was trained on, so a **retrain** comes with it.
Not a concern for the lab viz; flagged so it is not forgotten.

## Components & data flow

```
candidate tubes (unchanged) ──▶ stabilize.tube_window(tube) ──▶ (cx,cy,w,h) fixed window
                                                                      │
full frames ──────────────────────────────────────────────────────▶ film-strip view
```

### `src/tube_builder_lab/stabilize.py` (new, pure, unit-tested)

- `tube_window(tube: Tube, margin: float = MARGIN) -> tuple[float, float, float, float]`
  - Enclosing box of all **observed** detections (`e.detection is not None`) in
    the tube, expanded by `margin`, returned as normalized `(cx, cy, w, h)`.
  - Observed-only is sufficient: interpolated gap boxes are lerps of observed
    boxes, so they already fall inside the union of observed boxes.
- `MARGIN` module constant (default `1.3`) — tunable by editing the file, the
  same iterate-and-save-on-reload workflow as `candidate.py`.

### `viz.py`

- `stabilized_crop(image_path, window)` — squares + crops + resizes the fixed
  window, reusing the existing `norm_bbox_to_pixel_square` + `crop_and_resize`
  (matches model-input squaring). The margin lives in `tube_window`, so this
  helper does **not** re-apply `CROP_CONTEXT`.

### `app.py` — film-strip view

A new expander, **"stabilized vs per-frame (film strip)"**. For each candidate
tube, two thumbnail rows spanning the tube's `[start_frame, end_frame]`:

- **per-frame** (top): crop centered on each frame's own box — current jumpy
  behavior (reuses `crop_tube_at_frame`).
- **stabilized** (bottom): the single `tube_window` cropped from every frame.

Rendered as two `st.image([...])` calls (a list lays out as a horizontal strip),
small fixed thumbnail width (~90px), labeled with the tube color/id.

The existing "candidate crops @ this frame" expander is **left as-is**; the film
strip is additive.

## Edge cases

- **Gap / no-detection frame:** the stabilized row still crops the fixed window
  (the frame image exists), so motion stays continuous; the per-frame row uses
  the interpolated box if present, else a blank placeholder so the two rows stay
  column-aligned.
- **Single-box tube:** union = that box; window = box + margin.
- **Image edges:** `norm_bbox_to_pixel_square` already clamps to the image and
  pads to square — no special handling.

## Tests & tuning

- `tests/test_stabilize.py`: `tube_window` correctness — two boxes → enclosing
  union; margin scaling; single-box tube; values stay normalized/clamped. Pure
  function, no Streamlit.
- No new config/params; `MARGIN` is a constant tuned in-file, seen live on save.
- `make lint` + `make test` green.

## Out of scope

- Lifting `tube_window` into `bbox_tube_temporal/model_input.py` (approach B).
- Retraining the temporal head on stabilized crops.
- Smoothed/EMA windows and size-only variants (rejected during brainstorm).

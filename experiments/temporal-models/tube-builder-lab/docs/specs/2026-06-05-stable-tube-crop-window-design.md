# Per-tube stable crop window + comparison viz

Date: 2026-06-05
Status: implemented

## Problem

The crop the temporal head sees is computed **per-frame**. In
`bbox_tube_temporal` (`crop_tube_patches` / `model_input.process_tube`), each
tube entry's own bbox is expanded by `context_factor`, squared, and resized to
224. As a plume grows and drifts across the 30s-apart frames, the crop
**recenters and rescales every frame**: the smoke stays roughly centered and
same-size while the *background* slides and zooms. The result is a "jumpy"
sequence — the temporal head never sees the smoke actually move.

We want the opposite: a **single fixed crop window per tube**, applied to every
frame, so the background stays static and the smoke visibly moves and grows
inside it.

## Key decisions

- **What is fixed:** position *and* size — one fixed window for the whole tube
  (not size-only, not a smoothed/EMA window). Strongest "stable image, moving
  smoke" effect.
- **How the window is derived:** the **union (enclosing) box** of all the tube's
  observed detections. Guarantees the smoke is always inside the crop. Early
  frames show a small plume in a large window — acceptable and expected.
- **No separate margin.** An earlier draft expanded the union by its own
  `MARGIN` knob. Dropped: the crop step already adds context (`context_factor`,
  `viz.CROP_CONTEXT`), and a second margin both double-counted and made the
  stabilized crop tighter than the per-frame crops. The stabilized crop is now
  built **identically** to a per-frame crop — same `CROP_CONTEXT`, same squaring —
  just with the union box instead of a per-frame box. `CROP_CONTEXT` is the single
  shared context knob; it currently sits at **1.6**, tightened by eye in the lab
  below the model's training `context_factor ≈ 2.0` (a deliberate exploration
  choice — note it now lowers the non-stabilized reference crop too, so that crop
  no longer exactly mirrors the model's per-frame crop).
- **Placement (approach A → B):** stabilization is a **crop-window decision**,
  not a tube-*building* step. It does not change which detections link into which
  tube, so it never mutates the `Tube`.
  - **A (this spec):** lab-only pure function + comparison viz; no changes to
    `bbox_tube_temporal`.
  - **B (later follow-up):** lift `tube_window` into the lib crop path as the
    real temporal-head crop. Out of scope here.
- **Viz:** compare **non-stabilized vs stabilized** for the committed builder.
  The former (current) builder was removed from the app (the candidate is now the
  only baseline).

### Note on the eventual production step (B)

When stable crops are eventually fed to the temporal head, that is a different
input distribution than the head was trained on, so a **retrain** comes with it.
Not a concern for the lab viz; flagged so it is not forgotten.

## Components & data flow

```
candidate tubes (unchanged) ──▶ stabilize.tube_window(tube) ──▶ (cx,cy,w,h) union window
                                                                      │
full frames ──▶ viz.crop_box_px (expand by CROP_CONTEXT, square) ─────┴──▶ crop / drawn box
```

### `src/tube_builder_lab/stabilize.py` (pure, unit-tested)

- `tube_window(tube: Tube) -> tuple[float, float, float, float]` — the union
  (enclosing) box of all **observed** detections, normalized. Raises
  `ValueError` if the tube has no observed detection. Gap entries are ignored
  (interpolated gap boxes are lerps of observed boxes, so the observed union
  already encloses them). No margin applied.

### `src/tube_builder_lab/viz.py`

- `crop_box_px(bbox, img_w, img_h)` — **single source of truth** for the crop
  region: expand by `CROP_CONTEXT`, then `norm_bbox_to_pixel_square`. Used by both
  the crop and the drawn box, so *the box you see equals the crop you get*, for
  per-frame boxes and union windows alike.
- `crop_tube_at_frame(image_path, bbox)` — square 224 crop via `crop_box_px`.
  Pass a per-frame detection box for the non-stabilized crop, or a tube's
  `tube_window` for the stabilized crop — same code path.
- `draw_tube_windows(image_path, windows)` — draws each `(window, colour)` as its
  `crop_box_px` region; the stabilized-side mirror of `draw_tube_bboxes`, in the
  tube colour.

### `app.py`

- **Former builder removed everywhere:** no current-vs-candidate frame view,
  metric, timeline, or `current`/`Δ` columns; the summary tables show a single
  `tubes` count and grey out missing-cache rows.
- **Main two-up viewer** (driven by the play/slider): left = non-stabilized full
  frame with the per-frame box (`draw_tube_bboxes`); right = full frame with the
  fixed union window (`draw_tube_windows`). Both in the tube colour, both shown
  only for tubes active at the current frame; the window is identical every frame.
- **"candidate crops @ this frame"** expander: per tube, non-stabilized crop vs
  stabilized crop side by side, shown only when the tube is active at the frame.

## Edge cases

- **Gap / no-detection frame:** the tube is treated as inactive there — neither
  box nor crop is drawn (matches `draw_tube_bboxes` / `bboxes_at_frame`).
- **Single-box tube:** union = that box.
- **Image edges:** `norm_bbox_to_pixel_square` clamps to the image; `crop_and_resize`
  re-squares by padding the missing side.

## Tests

- `tests/test_stabilize.py`: `tube_window` — two-box union, single-box,
  axis-independent (w≠h) union, gap entries ignored, empty tube raises.
- `tests/test_viz.py`: `crop_box_px` — square + `CROP_CONTEXT`-scaled + centred,
  and edge clamping.
- `make lint` + `make test` green.

## Out of scope

- Lifting `tube_window` into `bbox_tube_temporal` (approach B).
- Retraining the temporal head on the stabilized-crop distribution.
- Smoothed/EMA windows and size-only variants (rejected during brainstorm).

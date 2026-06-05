# Stabilized per-tube crops + retrain (vit_dinov2_finetune A/B)

Date: 2026-06-05
Status: approved (brainstorm)

## Problem

The temporal head is fed **per-frame** crops: each tube entry's own bbox is
expanded by `context_factor` (currently 1.5), squared, and resized to 224
(`bbox_tube_temporal.model_input.process_tube` for training data,
`bbox_tube_temporal.inference.crop_tube_patches` at inference). Because the crop
recenters and rescales every frame, the smoke stays roughly centered/same-size
and the background slides — a "jumpy" sequence in which the smoke never appears
to move.

We prototyped an alternative in `tube-builder-lab`: a **single fixed crop window
per tube** (the union of the tube's observed boxes), applied to every frame, so
the background is static and the smoke visibly moves/grows inside it. This spec
incorporates that into the real experiment and **retrains `vit_dinov2_finetune`**
to measure whether it improves or degrades performance.

## Goal

Add an opt-in `stabilize` crop mode to the lib (training + inference), wire a
`model_input.stabilize` param through the experiment, and run a single stabilized
A/B against the **already-recorded** `vit_dinov2_finetune` baseline.

## Key decisions

- **A/B subject:** `vit_dinov2_finetune` only.
- **Baseline:** the committed `vit_dinov2_finetune` metrics
  (`data/08_reporting/val/vit_dinov2_finetune/`) — **not re-run**.
- **One arm, one seed:** a single stabilized run, seed 42, `context_factor=1.5`,
  merged tubes unchanged — so the only difference vs the recorded baseline is
  per-frame box → fixed-union box.
- **Mechanism:** `dvc exp run -S model_input.stabilize=true` targeting the dinov2
  eval — isolated experiment, committed baseline data/metrics untouched; compare
  with `dvc exp show`.
- **Window:** the union (enclosing) box of the tube's observed (non-gap,
  real-detection) boxes. No separate margin — the crop step's `context_factor`
  supplies the context (same value, 1.5, both arms).
- **Judgement:** holistic over the full protocol report — precision / recall /
  F1 / FPR, PR-AUC, ROC-AUC, and time-to-detection (TTD).
- **Implementation shape:** a `stabilize: bool` threaded through the two crop
  functions (rejected: a pluggable strategy object — YAGNI; precomputing the
  window onto the tube record — spreads the concept for no gain).

### Honest caveat (recorded with the result)

Single-seed, baseline-not-re-run. The val set is small (n=284; 7 FP, 1 FN), so
differences within seed noise are **not** conclusive. A small delta is "neutral,"
not a win/loss. Escalating to multi-seed (42/43/44, both arms) is the follow-up
if the result is borderline and worth pursuing.

## Components

### Lib — `lib/bbox-tube-temporal/src/bbox_tube_temporal/`

- `tube_window(tube) -> (cx, cy, w, h)` — union of the tube's observed
  detections, normalized; raises on a tube with no observed detection; ignores
  gap/interpolated entries (lerps of observed boxes, already enclosed). Ported
  from `tube-builder-lab`'s `stabilize.tube_window`. Pure, unit-tested.
- `process_tube(..., stabilize: bool = False)` — when true, compute the window
  once and crop that fixed box (`expand_bbox(window, context_factor)` → square →
  resize) for every frame. `False` is byte-identical to today.
- `crop_tube_patches(..., stabilize: bool = False)` — the same switch on the
  inference path, so a stabilized-trained model crops consistently when deployed.

### Experiment — `bbox-tube-temporal`

- `params.yaml`: add `model_input.stabilize: false` (new DVC param beside
  `context_factor` / `patch_size`).
- `scripts/build_model_input.py`: add `--stabilize` flag, pass to `process_tube`.
- `dvc.yaml` `model_input` stage: pass `${model_input.stabilize}` and list it
  under `params:` so toggling it invalidates the crop cache.
- **Inference parity:** `package_model` serializes `model_input.stabilize` into
  the packaged config; `BboxTubeTemporalModel.predict` passes it to
  `crop_tube_patches` (beside `mi["context_factor"]`, `model.py:225`). Default
  `false` keeps every existing packaged model valid.

## Data flow

```
tubes JSON ─┬─ build_model_input (--stabilize) ─▶ stabilized 224 PNG patches ─▶ train vit_dinov2_finetune ─▶ evaluate ─▶ metrics
            └─ tube_window (union) ── context_factor 1.5 ── same crop code as per-frame
packaged config { model_input.stabilize } ─▶ predict ─▶ crop_tube_patches(stabilize) ─▶ inference parity
```

## Testing

- Lib: `tube_window` (two-box union, single-box, gaps ignored, empty raises);
  `process_tube(stabilize=True)` yields a constant crop box across frames while
  `stabilize=False` is unchanged (tiny synthetic tube).
- Experiment: `--stabilize` reaches `process_tube`.
- `make lint` + `make test` green in **both** the lib and the experiment.

## Results (to fill after the run)

| metric | baseline (recorded) | stabilized | Δ |
|---|---|---|---|
| precision | 0.950 | | |
| recall | 0.993 | | |
| F1 | 0.971 | | |
| FPR | | | |
| PR-AUC | 0.991 | | |
| ROC-AUC | 0.993 | | |
| TTD (median frames) | | | |

Verdict: _improve / neutral / degrade_ — with the single-seed caveat.

## Out of scope

- Retraining other variants (gru_*, vit_in21k, etc.).
- Re-running the baseline / multi-seed (follow-up if borderline).
- Tuning `context_factor` or the window definition (held fixed to isolate the
  per-frame-vs-union change).
- Smoothed/EMA or size-only windows (rejected in the lab brainstorm).

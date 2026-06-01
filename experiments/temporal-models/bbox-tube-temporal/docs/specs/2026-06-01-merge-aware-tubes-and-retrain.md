# Merge-aware tubes in the lib + retrain `vit_dinov2_finetune` — design

**Date:** 2026-06-01
**Status:** approved (brainstorm)
**Follows:** the lab work in PR #72 (`experiments/temporal-models/tube-builder-lab`).

## Problem

The smoke-tube linking in `lib/bbox-tube-temporal/.../tubes.py::build_tubes` (greedy
1-to-1 IoU, `iou_threshold=0.2`, `max_misses=2`) over-fragments plumes across 30 s
frames. The lab developed and validated a **post-hoc co-located merge** that fuses
fragments of the same plume back together (`merge_colocated_tubes` in
`experiments/temporal-models/tube-builder-lab/.../candidate.py`). All 15 working-set
targets merge correctly; 17 reviewed control sequences match.

This spec propagates that merge into the shared lib and retrains the production
variant `vit_dinov2_finetune` so the classifier sees the new tube structure.

## Goals

- Make `merge_colocated_tubes` (and its helpers) a first-class part of
  `lib/bbox-tube-temporal`, with the same unit-test coverage as the lab has today.
- Make the merge **configurable through the packaged model config** — present on
  the new model, absent on existing models (backward compatibility).
- Regenerate training tubes through the new pipeline and **retrain
  `vit_dinov2_finetune`** so the classifier is trained on merged tubes.
- Evaluate the retrained model and report the delta against the current packaged
  one.

## Non-goals (this PR)

- Retraining other variants (`gru_convnext_finetune`, `mean_pool`, the other
  `vit_*`/`gru_*` baselines). They stay on the old tubes as historical baselines.
- Updating `temporal-model-explorer` or `temporal-model-leaderboard`.
- Tweaking the merge thresholds. The lab's tuned defaults (`merge_iomin=0.3`,
  `merge_prox_factor=1.0`, `merge_max_gap=10`) carry over unchanged.

## Lib changes (`lib/bbox-tube-temporal`)

### `src/bbox_tube_temporal/tubes.py`

Add `merge_colocated_tubes` plus the small helpers it composes
(`_same_plume`, `_time_gap`, `_closest_in_time`, `_same_box`, `_iou_min`,
`_center_distance`, `_combine`, `_connected_components`, `_box_rank`, `_observed`,
`_area`) — the refactored, top-down version from the lab. Keep `build_tubes` **pure
(greedy IoU linking only)** — the merge is a separate, optional pass. Export
`merge_colocated_tubes` from the package's `__init__.py`.

### `src/bbox_tube_temporal/model.py::predict`

The inference pipeline currently runs:

```
build_tubes → filter_and_interpolate_tubes(interpolate=True) → score
```

becomes:

```
build_tubes
  → filter_and_interpolate_tubes(interpolate=False)
  → merge_colocated_tubes              # only if merge keys present in config
  → filter_and_interpolate_tubes(interpolate=True)
  → score
```

Filter twice because the merge can change tube lengths/observation counts; the
second pass also performs the interpolation the classifier expects.

### Packaged config schema (`config["tubes"]`)

Add three optional keys:

```yaml
tubes:
  iou_threshold: 0.2
  max_misses: 2
  infer_min_tube_length: 2
  min_detected_entries: 2
  interpolate_gaps: true
  # New (optional). Present -> merge runs with these thresholds. Absent -> skip.
  merge_iomin: 0.3
  merge_prox_factor: 1.0
  merge_max_gap: 10
```

`model.py::predict` treats missing merge keys as "no merge" — existing packaged
models keep their exact current behavior (their classifier was trained on
unmerged tubes; running merged tubes through it would mismatch).

### Tests

Mirror the lab's `tests/test_candidate.py` into `lib/bbox-tube-temporal/tests/test_tubes.py`
(merge contained, keep distinct far apart, gap re-detection within window,
do-not-bridge beyond window, transitive, proximity scale-relative,
combine-tiebreak-deterministic). Plus a `test_model_edge_cases.py` (or
`test_model_parity.py`) addition asserting `predict` is bit-for-bit unchanged on
a config without the merge keys (backward compat).

## Experiment changes (`experiments/temporal-models/bbox-tube-temporal`)

### `params.yaml`

Extend the `tubes:` block:

```yaml
tubes:
  iou_threshold: 0.2
  max_misses: 2
  merge_iomin: 0.3
  merge_prox_factor: 1.0
  merge_max_gap: 10
```

### `scripts/build_tubes.py`

Apply `merge_colocated_tubes` after the existing build + filter (so the
serialized training tubes have the new linking). New CLI args:
`--merge-iomin`, `--merge-prox-factor`, `--merge-max-gap`. When any is omitted
(or set to `null`), the merge is skipped (preserves the legacy path).

### `dvc.yaml`

Update the `build_tubes` stage to:
- pass the new CLI args from `${tubes.*}`
- list the new params under `params:` (so DVC re-runs the stage when they change)
- add `src/bbox_tube_temporal/tubes.py` to `deps:` (already present via `data.py`
  but explicit is clearer)

Stage deps cascade: changing `build_tubes` invalidates `build_model_input`,
which invalidates `train_vit_dinov2_finetune`, which invalidates the rest of
its chain. DVC handles the rerun via `dvc repro`.

### `scripts/package_model.py` (and `src/bbox_tube_temporal/package.py`)

When building the model package, copy the merge keys from `params.yaml::tubes`
into the bundled `config.yaml` so `BboxTubeTemporalModel.from_package` reads
them.

### What stays the same

`dataset.py`, `train.py`, `evaluate.py`, the classifier, augmentation — all
unchanged. The merge changes the *tubes on disk*, not the training loop or the
model.

## Retrain + evaluation protocol

1. `dvc repro build_tubes` → new training tubes (train + val).
2. `dvc repro build_model_input` → new patch crops.
3. `dvc repro train_vit_dinov2_finetune` → new classifier checkpoint.
4. `dvc repro evaluate_vit_dinov2_finetune` (both `train` and `val`).
5. `dvc repro analyze_variant@vit_dinov2_finetune` → recommended_config + calibrator.
6. `dvc repro package@vit_dinov2_finetune` → new `model.zip` with merge config.
7. `dvc repro evaluate_packaged@{vit_dinov2_finetune,train|val}` → end-to-end metrics.

Compare new vs. current vit_dinov2_finetune metrics (precision/recall/F1 at the
logistic threshold; PR curve). The `compare_variants` step compares across
variants; we read the new vit_dinov2 metrics against the old metrics on `main`
(checked in as plots / `metrics.json`).

## Success criteria

- Lib tests pass (existing + new merge tests + the backward-compat parity test).
- Training pipeline reruns cleanly via `dvc repro` and produces a new
  `data/06_models/vit_dinov2_finetune/model.zip` whose bundled config contains
  the three merge keys.
- New `vit_dinov2_finetune` validation metrics are **not worse than** the old —
  at a minimum, recall@target ≥ current; precision/F1 not regressed beyond a
  small margin.
- Loading the *old* `model.zip` (no merge keys) yields identical `predict`
  output to today (parity test).
- The new model, when scored end-to-end on the lab's working-set sequences,
  shows the expected tube-count drops on the targets.

## Testing

- Unit tests in `lib/bbox-tube-temporal/tests/test_tubes.py` (the merge-logic
  set above).
- A backward-compatibility test in
  `lib/bbox-tube-temporal/tests/test_model_edge_cases.py` (or `test_model_parity.py`):
  a model config without merge keys produces the same `predict` outputs as
  before the change.
- Existing experiment tests (`test_data.py`, `test_dataset.py`, etc.) must
  still pass.
- A short end-to-end sanity check on a small synthetic dataset (script or
  pytest) confirming the full chain (build_tubes → … → predict) works with
  the merge keys present.

## Open questions

None blocking. If retraining shows a regression on `gru_convnext_finetune` is
also wanted, that becomes a follow-up PR (out of scope here).

# Stabilized `vit_dinov2_finetune` Variant — DVC Stage Design

**Date:** 2026-06-08
**Status:** Approved design, pending implementation plan.

## Goal

Add a permanent, reproducible **stabilized** `vit_dinov2_finetune` variant to the
bbox-tube-temporal pipeline that **coexists** with the existing per-frame variant.
The new variant runs the full chain — crop → train → package → evaluate → calibrate —
using stabilized crops (fixed per-tube union window) instead of the drifting
per-frame detection box.

This replaces the ephemeral `dvc exp run -S model_input.stabilize=true` workflow
(used for the one-off A/B) with first-class stages, so the stabilized model and its
metrics/calibrator are reproducible side-by-side with the per-frame arm.

## Background

The opt-in `stabilize` flag (merged in PR #82) is a crop-window decision threaded
through both crop paths (`process_tube`, `crop_tube_patches`), baked into the
packaged model config for inference parity, and wired through
`build_model_input --stabilize` from the global `model_input.stabilize` param
(default `false`).

The pipeline keys everything off a single global `model_input.stabilize` and one
crop dir (`data/05_model_input/`), with one explicitly-named stage per model
variant (`train_vit_dinov2_finetune`, `train_vit_in21k_finetune`, …). A coexisting
stabilized variant therefore needs its own crop dir, train, and package stages —
the global param stays `false` for the per-frame arm.

## Approach

**Parallel explicit `_stabilized` stages** that mirror the dinov2 path but read
stabilized crops. The per-frame pipeline and all other variants are untouched.

Rejected alternatives:
- **Convert package/train into a `stabilize`-matrix foreach** — DRY, but renames
  existing stages (`package@vit_dinov2_finetune` → `package@0`), invalidating the
  cache and the setup just merged via PR #82.
- **Separate experiment subproject (`cp -r`)** — total isolation but heavy
  duplication and drift.

## New artifacts

| Kind | Path |
|---|---|
| Crops | `data/05_model_input_stabilized/{train,val}` |
| Model | `data/06_models/vit_dinov2_finetune_stabilized/` (checkpoint + `model.zip`) |
| Packaged reports | `data/08_reporting/{train,val}/packaged/vit_dinov2_finetune_stabilized/` |
| Variant analysis | `data/08_reporting/variant_analysis/vit_dinov2_finetune_stabilized/` |

## Pipeline changes (`dvc.yaml`)

### 1. `build_model_input_stabilized` (new, foreach train/val)
Mirror `build_model_input`, but force `--stabilize true` and output to the
`_stabilized` crop dir.

```yaml
  build_model_input_stabilized:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/build_model_input.py
        --tubes-dir data/03_primary/tubes/${item}
        --raw-dir data/01_raw/datasets/${item}
        --output-dir data/05_model_input_stabilized/${item}
        --context-factor ${model_input.context_factor}
        --patch-size ${model_input.patch_size}
        --stabilize true
      deps:
        - scripts/build_model_input.py
        - ../../../lib/bbox-tube-temporal/src/bbox_tube_temporal/model_input.py
        - ../../../lib/bbox-tube-temporal/src/bbox_tube_temporal/stabilize.py
        - data/03_primary/tubes/${item}
        - data/01_raw/datasets/${item}
      params:
        - model_input.context_factor
        - model_input.patch_size
      outs:
        - data/05_model_input_stabilized/${item}
```

Note: `params` lists only the two keys this stage actually uses (`context_factor`,
`patch_size`) — not the whole `model_input` block — so toggling the global
`model_input.stabilize` does not needlessly invalidate the stabilized crop cache.

### 2. `train_vit_dinov2_finetune_stabilized` (new)
Identical to `train_vit_dinov2_finetune` except it reads the stabilized crop dir,
writes to the `_stabilized` model dir, and uses the aliased params key. Training
config is byte-identical to the per-frame variant — only the crops differ.

```yaml
  train_vit_dinov2_finetune_stabilized:
    cmd: >-
      uv run python scripts/train.py
      --arch transformer
      --train-dir data/05_model_input_stabilized/train
      --val-dir data/05_model_input_stabilized/val
      --output-dir data/06_models/vit_dinov2_finetune_stabilized
      --params-path params.yaml
      --params-key train_vit_dinov2_finetune_stabilized
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal_exp/augment.py
      - src/bbox_tube_temporal_exp/dataset.py
      - ../../../lib/bbox-tube-temporal/src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal_exp/lit_temporal.py
      - src/bbox_tube_temporal_exp/training_plots.py
      - data/05_model_input_stabilized/train
      - data/05_model_input_stabilized/val
    params:
      - train_vit_dinov2_finetune_stabilized
      - augment
    outs:
      - data/06_models/vit_dinov2_finetune_stabilized/best_checkpoint.pt
      - data/06_models/vit_dinov2_finetune_stabilized/csv_logs/
    plots:
      - data/06_models/vit_dinov2_finetune_stabilized/plots/training_curves.png
```

### 3. `package_vit_dinov2_finetune_stabilized` (new, explicit — not the shared foreach)
The shared `package` foreach `do.cmd` can't express per-variant args, so the
stabilized variant gets its own explicit stage passing `--stabilize true` and the
stabilized val crops for calibration.

```yaml
  package_vit_dinov2_finetune_stabilized:
    cmd: >-
      uv run python scripts/package_model.py
      --variant vit_dinov2_finetune_stabilized
      --stabilize true
      --val-patches-dir data/05_model_input_stabilized/val
      --output data/06_models/vit_dinov2_finetune_stabilized/model.zip
    deps:
      - data/06_models/vit_dinov2_finetune_stabilized/best_checkpoint.pt
      - data/01_raw/models/best.pt
      - data/05_model_input_stabilized/val
      - data/01_raw/datasets/train
      - data/01_raw/datasets/val
      - scripts/package_model.py
      - ../../../lib/bbox-tube-temporal/src/bbox_tube_temporal/package.py
      - src/bbox_tube_temporal_exp/calibration.py
      - src/bbox_tube_temporal_exp/val_predict.py
      - src/bbox_tube_temporal_exp/package_predict.py
      - ../../../lib/bbox-tube-temporal/src/bbox_tube_temporal/logistic_calibrator.py
      - src/bbox_tube_temporal_exp/logistic_calibrator_fit.py
      - ../../../lib/bbox-tube-temporal/src/bbox_tube_temporal/inference.py
      - ../../../lib/bbox-tube-temporal/src/bbox_tube_temporal/model.py
    params:
      - package.target_recall
      - package.infer
      - package.infer_min_tube_length
      - package.aggregation.vit_dinov2_finetune_stabilized
      - tubes
      - build_tubes
      - model_input
      - train_vit_dinov2_finetune_stabilized
    outs:
      - data/06_models/vit_dinov2_finetune_stabilized/model.zip

The `package.*` params are **scoped** (dotted) rather than tracking the whole
`package` block. The same scoping is applied to the existing shared `package`
foreach (`package.aggregation.${item}`) so that adding this variant's aggregation
rule does not ripple the per-frame package stages' param hash. `package_model.py`
reads exactly these four `package.*` keys (`target_recall`, `infer`,
`infer_min_tube_length`, `aggregation.<variant>`).
```

### 4. `evaluate_packaged` (append two foreach entries)
The model.zip bakes `stabilize: true`, so inference crops stabilized automatically
(no script change). Append:

```yaml
      - {variant: vit_dinov2_finetune_stabilized, split: train}
      - {variant: vit_dinov2_finetune_stabilized, split: val}
```

### 5. `analyze_variant` (append one foreach entry)
No script change; reads the stabilized packaged predictions.

```yaml
      - vit_dinov2_finetune_stabilized
```

## Code change (only one): `scripts/package_model.py`

Add a `--stabilize` flag (parsed like `build_model_input.py`'s `_to_bool`) threaded
into `_model_input_config` so the stabilized `model.zip` bakes `stabilize: true`.
Today `_model_input_config` always reads the global `model_input.stabilize`
(= `false`), which would mis-bake the stabilized variant and break inference parity.

```python
def _model_input_config(all_params: dict, stabilize: bool | None = None) -> dict:
    mi = all_params["model_input"]
    return {
        "context_factor": mi["context_factor"],
        "patch_size": mi["patch_size"],
        "stabilize": mi.get("stabilize", False) if stabilize is None else stabilize,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
```

Plumb a `stabilize` argument through `_build_config` → `_model_input_config`, add
`--stabilize` (default `None` → use the param) to the arg parser, and pass it.
`--val-patches-dir` already exists. Add a unit test asserting the baked config
carries `stabilize: true` when the flag is passed and falls back to the param when
it is not.

## `params.yaml` additions

Anchor the existing dinov2 block and alias it (zero hyperparam drift):

```yaml
train_vit_dinov2_finetune: &train_vit_dinov2_finetune
  <<: *vit_defaults
  backbone: vit_small_patch14_dinov2.lvd142m
  finetune: true
  finetune_last_n_blocks: 1
  backbone_lr: 0.00001

train_vit_dinov2_finetune_stabilized: *train_vit_dinov2_finetune
```

Add the variant to the per-variant decision-rule map:

```yaml
  aggregation:
    gru_convnext_finetune: max_logit
    vit_dinov2_finetune: logistic
    vit_dinov2_finetune_stabilized: logistic
```

## Per-frame impact

The per-frame variant's **behaviour and outputs are unchanged** — `model_input.stabilize`
stays `false`, `package_model.py` selects its aggregation rule per-variant
(`aggregation.get(variant)`), and packaging without `--stabilize` falls back to the
param, so per-frame `model.zip`s are byte-equivalent.

Two shared dependencies are edited, however, so the per-frame package stages will
**re-validate** (not change behaviour) on the next `dvc repro`:

- `scripts/package_model.py` gains `--stabilize` (backward-compatible). It is a
  legitimate dep of every `package` stage, so editing it re-hashes them — unavoidable
  for any shared-script change.
- The shared `package` foreach `params` are scoped from the whole `package` block to
  `package.target_recall`, `package.infer`, `package.infer_min_tube_length`,
  `package.aggregation.${item}` — so adding this variant's aggregation rule no longer
  ripples sibling stages going forward.

`build_model_input`, `train_vit_dinov2_finetune`, `evaluate_vit_dinov2_finetune`, and
the gru training stages are otherwise unchanged. No data regeneration is part of this
change beyond building the new variant's own stages.

## Testing

- Unit test for `package_model.py --stabilize` (baked config carries the flag;
  falls back to the param when absent).
- `dvc status` / dry parse confirms the new stages are valid and wired (without
  running the heavy repro).
- Existing lib + experiment suites stay green.

## Reproduction (post-merge, separate step)

```bash
uv run dvc repro analyze_variant@vit_dinov2_finetune_stabilized
```

reproduces the full stabilized chain (stabilized crops → train → package →
evaluate train+val → analyze) without touching the per-frame variant.

# Performance Improvements (Backbone + Fine-tuning) — Design

**Date:** 2026-04-14
**Status:** Draft, awaiting user review
**Scope:** `experiments/temporal-models/smokeynet-adapted/`

## Goal

Reduce false-positive count at a target recall of 0.95–0.99 on the
current per-tube binary classifier (`TemporalSmokeClassifier`). Today's
GRU variant sits at 17 FPs / 149 negatives at recall 0.993, and the
PR curve is flat in the 0.95–0.99 recall band — threshold tuning alone
does not move the operating point meaningfully. The ceiling is in the
per-frame features, not the temporal head.

## Non-goals

- No changes to the tube construction or model-input pipeline
  (`build_tubes`, `build_model_input` stages).
- No changes to inference (`model.py`) or packaging (`package.py`).
- No new data augmentation.
- No transformer temporal head.
- No hard-negative mining or site-aware resampling (flagged as
  follow-up; the 17 FPs cluster strongly in 4 cameras, which is worth
  revisiting after this spec lands).

## Approach

Three ordered experiments, each moving one axis at a time, plus
noise-floor seed reruns so that deltas on the small (284-sample) val
set can be interpreted.

1. **Exp 1 — `train_gru_convnext`**: swap the frozen `resnet18`
   backbone for a frozen `convnext_tiny`. Tests whether better-frozen
   features are enough.
2. **Exp 2 — `train_gru_finetune`**: keep `resnet18`, partially
   unfreeze (last block only) with a low backbone LR. The hero run.
3. **Exp 3 — `train_gru_convnext_finetune`** (conditional): combine 1 +
   2. Only runs if both Exp 1 and Exp 2 show signal above the
   noise floor.

Parallel: `train_gru_seed43` and `train_gru_seed44` — identical to
`train_gru` except for seed. Establishes the seed-to-seed spread on
the small val set so we can decide which deltas are real.

**Go/no-go for Exp 3** is a human judgment call on the comparison
report, not automated. "Signal above noise floor" is a prose rule:
the variant must beat the baseline mean by more than the seed-to-seed
spread on FP count at target recall.

## Config (`params.yaml`)

New sections, existing blocks untouched. Only the fields that *differ*
from `train_gru` are shown; all other knobs (epochs, weight_decay, etc.)
match `train_gru` exactly.

```yaml
train_gru_seed43:
  # identical to train_gru
  seed: 43

train_gru_seed44:
  # identical to train_gru
  seed: 44

train_gru_convnext:
  backbone: convnext_tiny
  finetune: false

train_gru_finetune:
  backbone: resnet18
  finetune: true
  finetune_last_n_blocks: 1
  backbone_lr: 1.0e-5

train_gru_convnext_finetune:
  backbone: convnext_tiny
  finetune: true
  finetune_last_n_blocks: 1
  backbone_lr: 1.0e-5
```

**Design decisions embedded:**

- `finetune_last_n_blocks` as a single int (not a list of layer
  names). Keeps config backbone-agnostic; the backbone module
  translates `N` to the right family-specific layers.
- Separate `backbone_lr` field (not a multiplicative factor). Explicit
  is clearer for per-group optimizer construction.
- No augmentation or dropout knobs. Existing `weight_decay: 0.01`
  provides the only regularization. Single-axis experiments.

## Code changes

Four files touched. Minimal and localized.

### 1. `src/bbox_tube_temporal/temporal_classifier.py`

Replace `FrozenTimmBackbone` with `TimmBackbone`, respecting a
`finetune` flag. `FrozenTimmBackbone` is renamed, not kept as an
alias — callers update.

```python
class TimmBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
    ) -> None:
        ...
```

Behavior:

- `finetune=False`: all params `requires_grad=False`, inner model
  forced to `.eval()` regardless of parent `.train()` state, forward
  wrapped in `@torch.no_grad()`. Byte-for-byte equivalent to today's
  `FrozenTimmBackbone`.
- `finetune=True`: the last *N* blocks get `requires_grad=True`;
  everything else frozen. Forward runs without `no_grad`. `.train()`
  propagates normally so BatchNorm on unfrozen blocks updates.
- Block resolution per family:
  - `resnet*`: `N=1` unfreezes `layer4`; `N=2` adds `layer3`.
  - `convnext_*`: `N=1` unfreezes `stages[-1]`; `N=2` adds
    `stages[-2]`.
  - Other families: `NotImplementedError` with the model's stage
    names in the exception message. Silent fall-through to
    "unfreeze everything" is explicitly avoided.

### 2. `src/bbox_tube_temporal/training.py`

`LightningModule.configure_optimizers` builds a per-group optimizer
when `finetune=True`:

```python
backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
head_params     = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone.")]
optimizer = AdamW([
    {"params": head_params,     "lr": head_lr,     "weight_decay": wd},
    {"params": backbone_params, "lr": backbone_lr, "weight_decay": wd},
])
```

When `finetune=False`, only the head has `requires_grad=True`, so a
single group at `head_lr` is built (behavior identical to today).

### 3. `dvc.yaml`

Five new `train_*` stages mirroring `train_gru`'s DVC definition, and
matching `evaluate_*_{train,val}` stages that reuse the existing
evaluate script with the new checkpoint paths. One new reporting
stage: `compare_variants` (see Evaluation).

### 4. `scripts/train.py`

Read `finetune`, `finetune_last_n_blocks`, `backbone_lr` from the
active variant's params block and thread them through to
`TemporalSmokeClassifier` / the Lightning module. CLI unchanged; the
stage name still selects the params section.

### Not touched

`model.py`, `package.py`, `heads.py`, `net.py`, `backbone.py` (the
unused YOLO+RoI path), dataset, dataloader.

## Evaluation

Existing `evaluate.py` stage is reused unchanged. Each variant writes
to its own `data/08_reporting/{train,val}/<variant>/` directory.

### New stage: `compare_variants`

New script `scripts/compare_variants.py`, wired as a DVC stage whose
dependencies are every variant's `predictions.json` and whose output
is `data/08_reporting/comparison.md`.

The Markdown table has one row per variant and the following columns:

- `variant`
- `F1 @ 0.5`
- `PR-AUC`
- `ROC-AUC`
- `FP @ recall 0.90`
- `FP @ recall 0.95`
- `FP @ recall 0.97`
- `FP @ recall 0.99`

The four target recalls give a full view of the PR tradeoff rather
than committing to a single operating point up front; the decision of
which recall is the "right" target is left to the reader of the
report.

### Noise-floor interpretation

Called out in prose at the top of `comparison.md`, not enforced in
code: *a variant must beat the baseline mean by more than the
seed-to-seed spread on FP count at target recall to count as
signal.* The `train_gru`, `train_gru_seed43`, and `train_gru_seed44`
rows in the same table provide that spread.

## Testing

Extensions to the existing suite. No integration training tests — too
slow, covered by `dvc repro` in practice.

### `tests/test_temporal_classifier.py`

- `test_timm_backbone_frozen_equivalence` — `TimmBackbone(finetune=False)`
  is numerically equivalent to the prior `FrozenTimmBackbone` on the
  same input (regression guard for the rename).
- `test_timm_backbone_finetune_requires_grad_resnet18` — with
  `finetune=True, finetune_last_n_blocks=1`, exactly the params in
  `layer4` have `requires_grad=True`; earlier layers frozen.
- `test_timm_backbone_finetune_requires_grad_convnext_tiny` — same,
  verifying `stages[-1]` unfrozen and `stages[:-1]` frozen.
- `test_timm_backbone_finetune_unsupported_family_raises` — e.g.
  `vit_small_patch16_224` with `finetune=True` raises
  `NotImplementedError` with stage names in the message.
- `test_timm_backbone_finetune_train_mode_propagates` — under
  `finetune=True`, `.train()` leaves unfrozen blocks in training mode
  (BatchNorm stats update) while frozen blocks remain `.eval()`.

### `tests/test_training.py`

- `test_configure_optimizers_single_group_when_frozen` —
  `finetune=False` produces one param group at head LR.
- `test_configure_optimizers_two_groups_when_finetune` —
  `finetune=True` produces two groups with distinct LRs; every
  trainable param lands in exactly one group.

### `tests/test_compare_variants.py` (new)

- `test_fp_at_target_recall` — synthetic `predictions.json` input;
  `fp_at_recall(preds, 0.95)` returns the expected FP count.
- `test_markdown_table_shape` — running the script on two fake
  variants produces a table with one header row and two data rows;
  no exception on empty variant list.

## Follow-up (out of scope)

- Hard-negative focused oversampling from the recurring-FP sites
  (`brison_20`, `brison_290`, `brison_39`, `croix-augas_8`).
- Transformer temporal head. Revisit only if the experiments in this
  spec plateau.
- Data augmentation (user excluded from this spec; worth revisiting
  independently).

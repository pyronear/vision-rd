# ViT backbone + Temporal Transformer variants

**Status:** Design
**Date:** 2026-04-15
**Experiment:** `experiments/temporal-models/bbox-tube-temporal/`

## Motivation

The current experiment reaches its best val metric with a CNN-per-frame
(`convnext_tiny`, last block finetuned) and a GRU temporal head. We want to
test whether transformer-based architectures — both spatial (ViT) and
temporal (a small transformer encoder) — are competitive on this dataset.

Frames are 30 s apart and cameras are fixed, so motion cues are weak
relative to dense video. The hypothesis is that strong pretrained spatial
features (especially DINOv2 self-supervised weights) transfer well to smoke
imagery, and that a small temporal transformer can model the short
(≤ 20 frame) tube sequences at least as well as the GRU.

## Scope

Add three new variants to the existing `bbox-tube-temporal` experiment. No
new uv project; the existing `truncate → build_tubes → build_model_input`
pipeline and the deployment packaging are reused unchanged.

| Variant | Backbone | Temporal head | Pretraining | Finetune |
|---|---|---|---|---|
| `train_vit_dinov2_frozen` | ViT-S/14 | Temporal Transformer | DINOv2 SSL | no |
| `train_vit_dinov2_finetune` | ViT-S/14 | Temporal Transformer | DINOv2 SSL | last 1 block |
| `train_vit_in21k_finetune` | ViT-S/16 | Temporal Transformer | IN21k sup. | last 1 block |

Seed 42 for all three first-pass runs. Multi-seed variance bars are
deferred until a winner is identified.

## Architecture

Factorized: per-frame ViT → sequence of per-frame CLS tokens → Temporal
Transformer → pooled → sequence logit. Same shape as the current
CNN + GRU pipeline so the comparison isolates the backbone-and-head swap.

### Per-frame backbone

```python
backbone = timm.create_model(name, pretrained=True, num_classes=0, global_pool="token")
```

Both ViT-S/14 (DINOv2) and ViT-S/16 (IN21k) emit a 384-d CLS embedding, so
the temporal head is identical across variants.

Input flow: `(B, T, 3, 224, 224)` → reshape to `(B·T, 3, 224, 224)` → ViT
→ `(B·T, 384)` → reshape to `(B, T, 384)`.

Frozen mode sets `requires_grad=False` on all backbone parameters. Finetune
mode unfreezes the last transformer block (parity with
`gru_convnext_finetune`'s "last 1 block" convention).

### Temporal transformer head

Small encoder over the T frame tokens:

- Prepend a learnable `[CLS]` token (1 × 384) → input is `(B, T+1, 384)`.
- Add learned absolute positional embeddings, size `(max_frames + 1, 384) = (21, 384)`,
  keyed by position within the tube (not by absolute timestamp).
- Encoder: 2 layers, 6 heads, `d_model=384`, `ffn_dim=1536`, `dropout=0.1`,
  pre-norm (standard ViT-style).
- Key-padding mask for tubes shorter than `max_frames`, sourced from the
  existing padding machinery in the dataset.
- Classifier: `Linear(384, 1)` on the `[CLS]` output token.

Parameter count: ViT-S ≈ 22M + temporal head ≈ 2M. Comparable to
`convnext_tiny` (~28M) + GRU (~0.5M), so throughput should be similar.

### Deployment

The `ViTTemporalTransformer` classifier slots into the existing
`temporal_classifier.py` alongside `MeanPool` and `GRU`. `BboxTubeTemporalModel`
wrapping, `max_logit` cross-tube aggregation, calibration, and the
packaged-archive flow are all reused unchanged.

## Training & integration

### Reused unchanged

- Tube construction (`tubes.py`), patch cropping (`model_input.py`), dataset
  (`dataset.py`), augmentation pipeline (`augment.py`).
- Lightning module skeleton (`lit_temporal.py`), BCE-with-logits loss.
- Reproducibility mechanism (`L.seed_everything(..., workers=True)`,
  `Trainer(deterministic=True)`).
- Calibration (`calibration.py`) and packaging (`package.py`,
  `scripts/package_model.py`).

### Code changes

- `src/bbox_tube_temporal/temporal_classifier.py` — add
  `ViTTemporalTransformer` module. Factory dispatches on
  `model.backbone.name` / `model.head.name`.
- `params.yaml` — three new `train_vit_*` sections mirroring the existing
  per-variant structure (lr, batch size, epochs, early stopping, seed,
  backbone/head config, finetune flag).
- `dvc.yaml` — three new `train_vit_*` stages, matching `evaluate_vit_*`
  foreach-val stages. `compare_variants` auto-aggregates.

### Hyperparameters (starting point)

- Optimizer: AdamW, `weight_decay=0.05` (standard for ViTs).
- Learning rate: `1e-4` for the temporal head and finetuned ViT blocks;
  frozen ViT params stay at `0`. Cosine schedule, 5 % warmup.
- Batch size: 16 tubes (same as current).
- Epochs / early stopping: inherit current variants' patience.
- Seed: 42.

Known tuning knob: if finetune variants are unstable, drop the ViT-block LR
to `3e-5` (layer-wise LR decay is out of scope for this spec).

### Dependencies

`timm` is already a project dependency. DINOv2 weights are fetched from HF
by `timm` on first load; `timm` will be pinned in `pyproject.toml` /
`uv.lock` so the resolved weights are stable. No new deps.

## Testing

- `tests/test_temporal_classifier.py` — extended to cover
  `ViTTemporalTransformer`: forward-pass output shape; padding-mask is
  honored (output identical whether padded positions are zero-filled or
  random-filled); CLS-token gradient flows through to the linear head;
  frozen-backbone mode leaves all ViT params with `requires_grad=False`.
- `tests/test_reproducibility.py` — add a ViT variant to the seeded
  two-run equality check.
- `tests/test_model_parity.py` — extend train/inference parity coverage to
  the ViT classifier (bitwise-equal logits between Lightning
  `validation_step` and `BboxTubeTemporalModel.predict_sequence`).

## Success criteria

1. **Primary.** At least one ViT variant beats `train_gru` (resnet18 + GRU)
   on the headline val metric used by `compare_variants`.
2. **Secondary.** ViT variant(s) land within 1–2 F1 points of
   `train_gru_convnext_finetune`. Matching or beating it is a bonus.
3. **Tertiary.** `train_vit_dinov2_frozen` lands within 2 F1 points of the
   finetuned variants. If so, it becomes the preferred deployment target
   (cheaper, smaller delta checkpoint, no finetuning instability).
4. **Diagnostic (not pass/fail).** Temporal attention maps on positive
   sequences peak on later frames where smoke has developed — a sanity
   signal that the temporal transformer is using order information.

## Out of scope

- Joint space-time attention (TimeSformer / ViViT-style).
- Multi-seed variance runs — deferred until a winning variant is chosen.
- Larger backbones (ViT-B, ViT-L).
- Time-aware positional encoding conditioned on real frame timestamps.
- Layer-wise LR decay for finetuned ViT blocks.

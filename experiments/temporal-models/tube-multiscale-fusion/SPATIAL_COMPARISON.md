# Spatial-module comparison — tube-multiscale-fusion

This compares architectures for the **spatial module**: the per-tube encoder in
the local branch that turns each spatio-temporal tube `(3, T, h, w)` into one
vector. It is the local-branch counterpart of [TEMPORAL_COMPARISON.md](TEMPORAL_COMPARISON.md).

## Setup

To isolate the spatial module, all runs use the **local-only** setting
(`ablation_variant: no_temporal`): the global DINOv2 sequence branch is removed,
tubes are extracted (2×2 grid of 112×112 cells × 4-frame windows at stride 2 →
28 tubes/seq), each tube is embedded by the **encoder under test**, the tube
vectors go through the fusion module (self-attention + learnable-query
cross-attention) and an MLP head. Only the per-tube encoder varies. Same data,
seed (42), schedule, augmentation. Metrics on the **val split (280 tubes)**,
single seed.

> Because the global branch is removed, these are *local-only* numbers — well
> below the full two-branch model (F1 0.978). This study answers "which encoder
> is the best **spatial module**", not "which model is best overall". See the
> [ablation study](ABLATIONS.md) for the branch-level picture.

| Encoder | Description |
|---------|-------------|
| **Tubelet transformer** (baseline/current) | Conv3d tubelet embedding → `[CLS]` + joint spatio-temporal self-attention. |
| **3D ResNet** | Kinetics-400 **pretrained** `r3d_18` per tube. |
| **ViViT** | Conv3d tubelet embedding → **factorised** spatial-then-temporal self-attention (ViViT Model-3). |
| **ConvLSTM** | Per-frame 2D CNN stem → ConvLSTM recurrence over the tube's frames. |
| **TSM** | Temporal Shift Module: 2D CNN with temporal channel shifts → temporal mean pool. |

All encoders except 3D ResNet are trained from scratch.

## Results (val, 280 tubes)

| Encoder | F1 | Acc | Prec | Recall | PR-AUC | FP | FN | Params | GFLOPs/seq | ms/seq¹ | R@FPR=1% | @5% |
|---------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **3D ResNet** (Kinetics) | **0.9416** | **0.9429** | **0.9281** | 0.9556 | **0.9788** | **10** | 6 | 40.6 M | 604.2 | 14.1 | **0.526** | **0.941** |
| **ViViT** (factorised) | 0.8904 | 0.8857 | 0.8280 | 0.9630 | 0.9235 | 27 | 5 | 14.8 M | 54.9 | 4.0 | 0.119 | 0.704 |
| **Tubelet transformer** (baseline) | 0.8716 | 0.8643 | 0.8012 | 0.9556 | 0.9019 | 32 | 6 | 11.3 M | 30.5 | 2.1 | 0.193 | 0.563 |
| **ConvLSTM** | 0.8581 | 0.8429 | 0.7600 | **0.9852** | 0.8679 | 42 | **2** | 8.7 M | 72.7 | 3.5 | 0.215 | 0.370 |
| **TSM** | 0.8503 | 0.8429 | 0.7862 | 0.9259 | 0.8594 | 34 | 10 | 7.7 M | 27.4 | 2.8 | 0.148 | 0.444 |

(sorted by F1) ¹ Batch-1, single RTX 4090.

## Findings

### 1. The spatial encoder choice matters a *lot* — far more than the temporal one.

F1 spans **0.85 → 0.94 (≈9 points)** across encoders, versus the ~0.8-point
spread in the [temporal comparison](TEMPORAL_COMPARISON.md). The reason is
structural: in the local-only setting the per-tube encoder *is* the entire
feature extractor (there is no DINOv2 global branch to lean on), so its quality
dominates the result. This is the opposite regime from the temporal study,
where strong DINOv2 features made the aggregator a minor lever.

### 2. Pretraining is the dominant factor — 3D ResNet wins decisively.

`r3d_18` is the **only** pretrained encoder and it is far ahead: **F1 0.942,
PR-AUC 0.979, recall@5%FPR 0.941**, and it roughly **thirds the false positives**
(10 vs the baseline's 32). Kinetics-400 ships it with strong, general motion
priors that the from-scratch encoders cannot learn from ~2.8 k training tubes.
The cost is steep: **20× the FLOPs** (604 vs 30.5) and the most parameters
(40.6 M), since `r3d_18` runs over all 28 tubes per sequence.

### 3. Among from-scratch encoders, factorised attention (ViViT) is best.

Excluding the pretrained model, the ranking is
**ViViT (0.890) > tubelet-transformer (0.872) > ConvLSTM (0.858) > TSM (0.850)**.
ViViT's factorised spatial-then-temporal attention edges the current joint
self-attention tubelet encoder by ~1.9 F1 at ~2× the FLOPs (54.9 vs 30.5) —
a modest, cheap upgrade.

### 4. ConvLSTM and TSM are not worth it here.

Both are the lightest (7–9 M params) but the **worst on F1 and precision**: they
over-fire (ConvLSTM recall 0.985 but precision 0.76, 42 FP). Recurrent/shift
convolutional encoders trained from scratch on this small dataset don't match
even the from-scratch transformers, and bring no compute advantage over the
baseline tubelet encoder.

### Recommendation

- **In isolation, `r3d_18` is the clear best spatial module** — pretraining is
  the decisive factor. Choose it if the local branch is to be a primary
  pathway, and budget for its 20× FLOPs.
- **For the actual two-branch model**, recall from the [ablation study](ABLATIONS.md)
  that the local branch adds only ~0.7 F1 on top of the global branch. So the
  expensive `r3d_18` is likely a poor trade there, and **ViViT** is the more
  sensible upgrade over the current tubelet-transformer: slightly better, still
  cheap, trained from scratch. Either change should be re-validated end-to-end
  in the full model before adoption.
- The current **tubelet-transformer** default is a reasonable
  cost/quality middle ground for the local branch.

## Caveats

- **Local-only numbers.** These isolate the spatial module and are not directly
  comparable to the full model; the best path is to re-run the top encoders
  (r3d_18, ViViT) inside the full two-branch model.
- **Single seed, small val set (280).** The from-scratch encoders cluster
  within run-to-run noise of each other (F1 0.85–0.89); the pretrained-vs-scratch
  gap and the precision differences are large and robust.
- `r3d_18` ran at batch size 8 (vs 16 for the others) for memory; this affects
  only optimisation noise, not the reported FLOPs/latency.

## Reproduce

```bash
# Tubelet-transformer baseline (local-only)
uv run dvc repro train_ablation_no_temporal evaluate_ablation_no_temporal
# Encoder variants
uv run dvc repro train_spatial evaluate_spatial
```

Per-variant metrics, confusion matrices, and PR/ROC curves are written to
`data/08_reporting/spatial_<enc>/{train,val}/` (and
`data/08_reporting/ablation_no_temporal/` for the tubelet-transformer baseline).

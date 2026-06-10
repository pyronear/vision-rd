# Temporal-module comparison — tube-multiscale-fusion

The [ablation study](ABLATIONS.md) showed the **temporal module** (the global
branch that aggregates the 16 per-frame DINOv2 embeddings into one sequence
vector) is by far the most important component. This document compares
different architectures for that aggregator.

## Setup

To isolate the temporal module, all runs use the **global-only** setting
(`ablation_variant: no_spatial`): DINOv2 ViT-S/14 per-frame (last block
fine-tuned) → **aggregator** → MLP head. No local tube branch, no fusion — so
the only thing that varies is how the `(B, 16, 384)` per-frame sequence is
collapsed to a single `(B, 384)` vector. Same data, seed (42), schedule, and
augmentation throughout. Metrics on the **val split (280 tubes)**, single seed.

The aggregator takes `(B, T, D)` + a frame mask and returns `(B, D)`:

| Aggregator | How it collapses the sequence |
|------------|-------------------------------|
| **Transformer** (baseline/default) | Learnable `[CLS]` + positional embeddings + 2-layer Transformer encoder; CLS readout. |
| **LSTM** | `nn.LSTM` over packed (masked) frames; last hidden state. |
| **GRU** | `nn.GRU` over packed (masked) frames; last hidden state. |
| **MLP** | Zero padded frames, flatten `T×D`, 2-layer MLP. |
| **1D CNN** | `Conv1d` over the temporal axis (2 layers, k=3) + masked mean pool. |
| **Linear + weighted avg** | Per-frame `Linear`, learned scalar score per frame, softmax-weighted average. |

## Results (val, 280 tubes)

| Aggregator | F1 | Acc | Prec | Recall | PR-AUC | FP | FN | Recall@FPR=1% | @5% |
|------------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LSTM** | **0.9747** | 0.9750 | 0.9507 | **1.0000** | 0.9930 | 7 | **0** | 0.867 | **1.000** |
| **MLP** | 0.9745 | 0.9750 | **0.9571** | 0.9926 | 0.9873 | 6 | 1 | 0.519 | 0.993 |
| **1D CNN** | 0.9710 | 0.9714 | 0.9504 | 0.9926 | 0.9897 | 7 | 1 | 0.770 | 0.993 |
| **Transformer** (baseline) | 0.9708 | 0.9714 | 0.9568 | 0.9852 | 0.9901 | 6 | 2 | 0.770 | 0.993 |
| **GRU** | 0.9673 | 0.9679 | 0.9500 | 0.9852 | **0.9952** | 7 | 2 | **0.889** | **1.000** |
| **Linear + weighted avg** | 0.9603 | 0.9607 | 0.9366 | 0.9852 | 0.9908 | 9 | 2 | 0.815 | 0.985 |

(sorted by F1)

### Size & compute

| Aggregator | Total params | Trainable | GFLOPs/seq | Latency (ms/seq)¹ |
|------------|---:|---:|---:|---:|
| Transformer | 25.3 M | 5.43 M | 196.1 | 8.8 |
| LSTM | 22.9 M | 3.06 M | 196.0 | 8.9 |
| GRU | 22.6 M | 2.76 M | 196.0 | 8.9 |
| MLP | 28.4 M | 8.56 M | 196.0 | 8.7 |
| 1D CNN | 22.6 M | 2.76 M | 196.0 | 8.7 |
| Linear + weighted avg | 21.9 M | 2.02 M | 196.0 | 8.8 |

¹ Batch-1, single RTX 4090.

## Findings

### 1. Compute is identical — the aggregator choice is essentially free.

Every variant lands at **~196 GFLOPs and ~8.8 ms/seq**. The 16 per-frame DINOv2
forward passes dominate so completely that the aggregator (0.15–6.7 M params)
is a rounding error in both FLOPs and latency. The decision is therefore purely
about **accuracy and parameter count**, not speed.

### 2. The architecture barely matters — every learned mixer lands in a tight band.

Five of the six aggregators sit within **F1 0.967–0.975** (≈0.8 points). Once
you have strong per-frame DINOv2 features, *how* you mix them across time is a
minor lever. The model is robust to this choice.

### 3. …but you do need real temporal mixing — pure weighted averaging is worst.

`Linear + weighted avg`, which has no frame-to-frame interaction (just a learned
average), is the clear laggard: lowest F1 (0.960) and most false positives (9).
Collapsing the sequence by a weighted mean discards the *order and dynamics* of
how the region evolves — exactly the turbulence-vs-drift signal the temporal
module exists to capture. Some sequential/interaction capacity is needed.

### 4. Recurrent models (GRU/LSTM) are the sweet spot.

- **GRU** gives the best ranking metrics — **PR-AUC 0.9952** and **recall@1%FPR
  0.889** (the deployment-critical low-false-alarm point) — with the *fewest*
  trainable params (2.76 M, half the Transformer's 5.43 M). Its lower F1 at the
  0.5 threshold (0.967) is a thresholding artefact, not a ranking weakness.
- **LSTM** gives the best raw F1 (0.975) and perfect recall (0 missed smokes),
  with R@5%FPR = 1.000.

Both recurrent models **edge the Transformer baseline** on this isolated
temporal task while being smaller.

### 5. MLP is a trap at the strict operating point.

The MLP ties for the best F1 at threshold 0.5 (0.975) but has by far the
**worst recall@1%FPR (0.519)** and the most trainable params (8.56 M). Its
scores rank poorly in the low-FPR tail — good "average" accuracy that would
fail a strict false-alarm budget. Flattening `T×D` also bakes in a fixed
sequence length and gives no real temporal inductive bias.

### Recommendation

For this architecture, a **GRU** aggregator is the best default: best ranking
(PR-AUC, recall@low-FPR), half the parameters of the Transformer, and identical
compute. **LSTM** is the alternative if raw F1 / recall is the target. The
current **Transformer** default is mid-pack — competitive but not the strongest,
and the heaviest of the recurrent/CNN options. Avoid **MLP** (poor low-FPR
behaviour) and **linear weighted-average** (no temporal mixing) for a
false-alarm-sensitive deployment.

> Note: this comparison is in the global-only setting. In the full two-branch
> model the cross-attention fusion may interact with the aggregator choice;
> swapping the production default to GRU should be re-validated end-to-end.

## Caveats

- **Single seed, small val set (280 tubes).** F1 gaps under ~0.005 (e.g.
  LSTM vs MLP vs 1D CNN) are within run-to-run noise. The recurrent-vs-average
  gap and the MLP low-FPR weakness are larger and more likely real, but a
  multi-seed sweep is needed before changing the production default.
- `recall@FPR=1%` is estimated from only ~145 negatives, so its tail is noisy.

## Reproduce

```bash
# Transformer baseline (global-only)
uv run dvc repro train_ablation_no_spatial evaluate_ablation_no_spatial
# Aggregator variants
uv run dvc repro train_temporal evaluate_temporal
```

Per-variant metrics, confusion matrices, and PR/ROC curves are written to
`data/08_reporting/temporal_<agg>/{train,val}/` (and
`data/08_reporting/ablation_no_spatial/` for the Transformer baseline).

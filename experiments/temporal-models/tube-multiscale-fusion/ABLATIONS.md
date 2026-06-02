# Ablation study — tube-multiscale-fusion

This study decomposes the full two-branch model by removing or simplifying one
component at a time, to attribute the model's performance to its parts. All
runs use the **same data, seed (42), augmentation, training schedule, and tube
geometry** (2×2 grid of 112×112 cells × 4-frame windows at stride 2 → 28 tubes
per sequence) as the default model; only the named component changes. Metrics
are on the **val split (280 tubes: 135 smoke / 145 fp)**, single seed.

## Variants

| # | Variant | What changed vs. full model |
|---|---------|------------------------------|
| 0 | **Full** | Global DINOv2 sequence branch + local tube branch + cross-attention fusion (self-attn over tubes, global=query). |
| 1 | **− temporal** (`no_temporal`) | **Global branch removed.** Local branch + fusion kept; the fusion's global query is replaced by a learnable token. |
| 2 | **− spatial** (`no_spatial`) | **Local branch removed.** Global branch only → MLP head; no fusion. |
| 3 | **fusion → weighted-mean** (`weighted_mean`) | Both branches kept; **cross-attention fusion replaced** by a learned weighted mean over tubes + a learned gate between the two branches (no attention). |

Each variant is a clean, single-component change: the kept modules are
byte-for-byte identical to the full model.

## Results (val, 280 tubes)

### Classification quality

| Variant | F1 | Accuracy | Precision | Recall | PR-AUC | FP | FN |
|---------|---:|---:|---:|---:|---:|---:|---:|
| **Full** | **0.9783** | 0.9786 | 0.9574 | **1.0000** | 0.9936 | 6 | 0 |
| fusion → weighted-mean | 0.9779 | 0.9786 | **0.9708** | 0.9852 | **0.9952** | **4** | 2 |
| − spatial (global only) | 0.9708 | 0.9714 | 0.9568 | 0.9852 | 0.9901 | 6 | 2 |
| − temporal (local only) | 0.8716 | 0.8643 | 0.8012 | 0.9556 | 0.9019 | 32 | 6 |

### Operating point & compute

| Variant | Recall@FPR=1% | @5% | @10% | Params | Trainable | GFLOPs/seq | Latency (ms/seq)¹ |
|---------|---:|---:|---:|---:|---:|---:|---:|
| **Full** | **0.874** | **1.000** | **1.000** | 36.4 M | 16.6 M | 226.5 | 18.9 |
| fusion → weighted-mean | 0.837 | 0.993 | 1.000 | 29.3 M | 9.5 M | 226.3 | 24.1 |
| − spatial (global only) | 0.770 | 0.993 | 0.993 | 25.3 M | 5.4 M | 196.1 | 20.6 |
| − temporal (local only) | 0.193 | 0.563 | 0.748 | 11.3 M | 11.3 M | 30.5 | 6.2 |

¹ Batch-1, single RTX 4090, one consistent measurement pass. Batch-1 latency is
dominated by kernel-launch overhead and is *not* strictly proportional to FLOPs
(hence `no_spatial` ≈ `full` despite fewer FLOPs); treat GFLOPs as the reliable
compute metric.

## Findings

### 1. The temporal (global) module is by far the most important component.

Removing it (`no_temporal`) is **catastrophic**: F1 collapses 0.978 → 0.872,
precision 0.957 → 0.801 (false positives jump 6 → 32), and recall at a strict
1 % FPR falls off a cliff, 0.874 → 0.193. Recall holds up best (1.000 → 0.956),
so the local branch alone still *finds* smoke — it simply cannot *reject*
slow-moving look-alikes (cloud, fog, haze) without the long-range sequence
context. The global branch is the workhorse; everything else refines it.

### 2. The spatial (local) module adds a modest but real gain — concentrated at the strict operating point.

Global-only (`no_spatial`) is already strong (F1 0.971), so the local branch
buys only ~0.7 F1 on average. But its value shows up where it matters for
deployment: **recall @ 1 % FPR improves 0.770 → 0.874** and PR-AUC 0.990 →
0.994 when the local branch is added back. The high-frequency turbulence cue
from the tubes sharpens the model's confidence precisely in the
low-false-alarm regime — consistent with the design rationale that local motion
detail disambiguates the hardest cases.

### 3. The cross-attention fusion is **not** clearly better than a simple weighted mean.

Replacing the attention-based fusion with a learned weighted mean
(`weighted_mean`) **matches the full model**: F1 0.9779 vs 0.9783 (a one-tube
difference, within noise), with *higher* precision (0.971 vs 0.957, 4 FP vs 6)
and *higher* PR-AUC (0.9952 vs 0.9936). It trails the full model only at the
very strictest 1 % FPR point (0.837 vs 0.874).

The takeaway: on this dataset the gain comes from **having both branches**, not
from the attention mechanism in the fusion step. Since the fusion is a
negligible fraction of compute (both variants are ~226 GFLOPs — the cost is
the 16 DINOv2 passes), the cross-attention buys a small edge at the extreme
operating point for ~7 M extra parameters. The weighted-mean fusion is a strong
**simplification candidate** if that extreme-FPR margin is not required.

### Summary

```
contribution to val F1 (full = 0.978):
  temporal/global module : +0.106   (0.872 → 0.978)   ← dominant
  spatial/local module   : +0.007   (0.971 → 0.978)   ← modest, helps low-FPR
  cross-attn over w-mean  : +0.000   (0.978 → 0.978)   ← negligible on this val set
```

The architecture's accuracy is driven overwhelmingly by the **global temporal
context**, with the **local tube branch** adding a small but deployment-relevant
gain at low false-alarm rates. The **attention-based fusion** is the most
expendable piece: a learned weighted mean is statistically indistinguishable
here and even slightly more precise.

## Caveats

- **Single seed, small val set (280 tubes).** Differences under ~0.005 F1
  (e.g. full vs. weighted-mean) are within run-to-run noise; the temporal and
  spatial effects are large enough to be robust, but the fusion comparison
  should be confirmed with multiple seeds before acting on it.
- All variants share the default 2×2 tube geometry; interactions between the
  fusion choice and finer/denser tube geometries were not explored.

## Reproduce

```bash
# Full model
uv run dvc repro train_dinov2_multiscale evaluate_dinov2_multiscale
# Ablations
uv run dvc repro train_ablation_no_temporal      evaluate_ablation_no_temporal
uv run dvc repro train_ablation_no_spatial       evaluate_ablation_no_spatial
uv run dvc repro train_ablation_weighted_mean    evaluate_ablation_weighted_mean
```

Per-variant metrics, confusion matrices, and PR/ROC curves are written to
`data/08_reporting/<variant>/{train,val}/`.

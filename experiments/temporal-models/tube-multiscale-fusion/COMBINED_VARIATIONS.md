# Combined spatial × temporal variations — tube-multiscale-fusion

The [spatial](SPATIAL_COMPARISON.md) and [temporal](TEMPORAL_COMPARISON.md)
comparisons each varied one module *in isolation* (local-only / global-only).
This study puts both modules back together and sweeps the **cartesian product**
of the strongest candidates **inside the full two-branch model** (global DINOv2
branch + local tube branch + cross-attention fusion + head):

- **spatial** (local per-tube encoder): `tubelet transformer`, `ViViT`, `3D ResNet`
- **temporal** (global aggregator): `transformer`, `LSTM`

= 6 full models. Same data, seed (42), schedule, augmentation, and 2×2 tube
geometry; only the two module kinds change. `tubelet × transformer` is the
production default (= `dinov2_multiscale`). Metrics on the **val split
(280 tubes)**, single seed.

## Results (val, 280 tubes)

| Spatial | Temporal | F1 | Acc | Prec | Recall | PR-AUC | FP | FN | Params | GFLOPs | ms/seq¹ | R@FPR=1% | @5% |
|---------|----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **tubelet** | **transformer** (default) | **0.9783** | 0.9786 | 0.9574 | **1.0000** | 0.9936 | 6 | **0** | 36.4 M | 226.5 | 10.8 | 0.874 | **1.000** |
| **resnet3d** | **LSTM** | 0.9779 | 0.9786 | **0.9708** | 0.9852 | **0.9952** | **4** | 2 | 63.4 M | 800.1 | 23.1 | 0.911 | 0.993 |
| vivit | LSTM | 0.9708 | 0.9714 | 0.9568 | 0.9852 | 0.9911 | 6 | 2 | 37.6 M | 250.9 | 12.8 | **0.933** | 0.993 |
| resnet3d | transformer | 0.9675 | 0.9679 | 0.9437 | 0.9926 | 0.9946 | 8 | 1 | 65.7 M | 800.2 | 22.9 | 0.867 | 0.993 |
| vivit | transformer | 0.9675 | 0.9679 | 0.9437 | 0.9926 | 0.9931 | 8 | 1 | 40.0 M | 251.0 | 12.6 | 0.793 | 0.993 |
| tubelet | LSTM | 0.9673 | 0.9679 | 0.9500 | 0.9852 | 0.9950 | 7 | 2 | 34.1 M | 226.4 | 10.9 | 0.874 | 0.985 |

(sorted by F1) ¹ Batch-1, single RTX 4090.

## Findings

### 1. In the full model, the spatial × temporal choice barely moves the needle.

All six combinations land within **F1 0.967–0.978 (~1.1 points)** — essentially
a tie on this val set. This is the key result and it contrasts sharply with the
isolated [spatial study](SPATIAL_COMPARISON.md), where the encoder choice spanned
9 F1 points (0.85–0.94). The explanation is the [ablation](ABLATIONS.md) finding:
once the strong **global DINOv2 branch** is present, it carries the prediction
and the local branch contributes only ~0.7 F1. So swapping the local encoder
(tubelet → ViViT → 3D ResNet) or the aggregator (transformer ↔ LSTM) only
reshuffles that small residual — within noise. **The architecture is robust to
these choices; none of the expensive upgrades is worth it in the full model.**

### 2. The cheapest model is also (statistically) the best.

`tubelet × transformer` — the current default and the **least expensive**
(226 GFLOPs, 10.8 ms, 36 M params) — has the top F1 (0.978), perfect recall
(0 missed smokes), and R@5%FPR = 1.000. Nothing beats it by more than noise, and
everything else costs more.

### 3. `resnet3d × LSTM` is the only combo that rivals it — at 3.5× the compute.

It edges the default on **precision (0.971 vs 0.957, 4 FP vs 6)** and **PR-AUC
(0.9952)**, i.e. slightly cleaner false-alarm behaviour. But it costs **800
GFLOPs (3.5×) and 23 ms (2.1×)** for a difference well inside single-seed noise.
Not a worthwhile trade unless false positives are extremely costly *and* the
compute budget is large.

### 4. Temporal aggregator: LSTM ≈ transformer; spatial encoder: pretraining helps precision, not F1.

- LSTM vs transformer is a wash across spatial backbones (LSTM wins for
  resnet3d/vivit, transformer wins for tubelet — all within noise), and the
  aggregator is compute-negligible (LSTM is marginally smaller).
- The pretrained `3D ResNet` spatial encoder shows up only as a small
  precision/PR-AUC edge (best with LSTM), **not** as an F1 win — consistent with
  the local branch being a minor contributor here. Its standalone dominance in
  the [spatial study](SPATIAL_COMPARISON.md) does **not** transfer to the full
  model, because the global branch already supplies most of the signal.

### Recommendation

**Keep the current default, `tubelet × transformer`.** It is the cheapest and is
statistically the best on this val set (top F1, perfect recall). If a deployment
is acutely false-alarm-sensitive and compute is not a constraint,
`resnet3d × LSTM` offers marginally higher precision/PR-AUC at 3.5× the FLOPs —
otherwise it is not justified. `vivit × LSTM` is a middle option (best
recall@1%FPR, +10% FLOPs) but no better on F1.

The broader lesson across all four studies: **the global temporal branch is the
workhorse; the local spatial branch and the specific module architectures are
second-order.** Spend complexity budget on the global branch, not on heavier
local encoders.

## Caveats

- **Single seed, small val set (280).** Every F1 gap here is < 0.012 — within
  run-to-run noise. The compute/precision differences are robust; the F1
  ranking is not. A multi-seed sweep would be needed to call a winner beyond the
  default.
- `tubelet × transformer` is reused from the existing `dinov2_multiscale` run
  (identical config). The two `resnet3d` combos trained at batch size 8 (others
  16) for memory; this affects optimisation noise only, not the reported
  FLOPs/latency.

## Reproduce

```bash
# Default cell (tubelet x transformer):
uv run dvc repro train_dinov2_multiscale evaluate_dinov2_multiscale
# The other 5 combos:
uv run dvc repro train_full_combo evaluate_full_combo
```

Per-combo metrics, confusion matrices, and PR/ROC curves are in
`data/08_reporting/full_<spatial>_<temporal>/{train,val}/` (and
`data/08_reporting/dinov2_multiscale/` for the default cell).

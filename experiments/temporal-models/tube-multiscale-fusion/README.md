# tube-multiscale-fusion

Two-branch temporal smoke classifier with multiscale spatio-temporal context.
Evolution of [`bbox-tube-motion-fusion`](../bbox-tube-motion-fusion/): drops the
per-frame motion CNN and adds a local spatio-temporal tube transformer that
captures high-frequency detail, while keeping a global DINOv2 sequence context
branch for long-range plume evolution.

## Motivation

Smoke and its hardest distractors — clouds, fog, haze, dust — can look almost
identical in any *single* frame. What separates them is **how they move**, and
crucially they move differently at different scales. This model is built around
that observation, which is why it splits the problem into a **global** and a
**local** branch instead of relying on one monolithic spatio-temporal encoder.

### Why two branches instead of one

A single encoder forced to describe a whole tube must trade off scales: pool
aggressively and you keep the long-range shape but blur out fine texture; keep
the texture and you blow up the token count and lose a clean notion of "the
overall trajectory." Smoke discrimination needs *both* signals at once, and
they live at genuinely different scales, so we model them with two specialised
branches and let a fusion step decide how to combine them.

- **Global branch — coarse shape over long time.** Each frame is embedded with
  DINOv2 and the 16-frame sequence is summarised into one context vector. This
  captures the *low-frequency* dynamics: is the region roughly static or slowly
  drifting (clouds, fog banks, a stationary haze layer), or is it growing,
  rising, and changing silhouette over many seconds (a developing plume)? This
  is the question best answered by looking at the entire sequence at once, and
  it is exactly the kind of smooth, large-scale evolution that survives heavy
  spatial pooling.

- **Local branch — high-frequency detail over short time.** The telltale sign
  of *smoke* rather than a cloud is **turbulence**: the boundary flickers,
  wisps detach, internal texture roils and reorganises from frame to frame.
  This is a *high-frequency* spatio-temporal signal, and it is precisely what a
  single global vector destroys — averaging the whole patch to 384 numbers
  smooths away the very flicker that distinguishes turbulent smoke from the
  laminar, low-frequency drift of fog or cloud. To preserve it we keep the
  patch decomposed into small spatial cells and track each cell across a short
  window of frames (a *tube*). Encoding each tube separately lets the model
  read the local motion signature — turbulent and high-variance for smoke,
  smooth and coherent for clouds/fog — that is invisible at the global scale.

### Why process *tubes* (local motion) at all

A static crop tells you what a region looks like; a tube tells you how that
region *changes*. By feeding the local encoder short, overlapping spatial-
temporal volumes we ask it to characterise per-cell motion rather than per-cell
appearance. Turbulent smoke produces strong, irregular frame-to-frame change
inside a cell; clouds and fog produce weak, slowly-varying change. Overlapping
the temporal windows (stride < tube length) means no transition between frames
falls on a window boundary, so short-lived turbulent events are always captured
inside at least one tube. This is the high-frequency cue that appearance-only
and globally-pooled models systematically miss, and it is why early, ambiguous
frames — where a nascent plume still *looks* like haze — can still be flagged
from their motion.

### Why fuse with global-as-query cross-attention

The two signals are complementary but asymmetric: the local tubes are the
*evidence* (where and how strongly is something moving turbulently?), and the
global vector is the *context* (over the whole sequence, is this consistent
with a growing plume or with static weather?). We therefore let the local tube
tokens first exchange information among themselves (self-attention, so a
turbulent cell can be read relative to its neighbours), then let the **global
vector act as the query** that attends over the local tubes (cross-attention).
This phrases the final decision as: *"given the overall trajectory I observed,
which local motion signatures matter, and do they add up to smoke?"* — using
long-range context to weight and interpret the high-frequency local evidence,
rather than averaging the two and hoping the signal survives.

The empirical payoff (see Results) is that this RGB-only design matches the
previous motion-CNN baseline **without** any precomputed optical-flow / frame-
difference inputs: the local tube transformer learns the turbulence cue
directly from pixels.

## Approach

```
patches (B, 16, 3, 224, 224) + mask (B, 16)
        │
        ├──► Global branch
        │       DINOv2 ViT-S/14 per frame  →  (B, 16, 384)
        │       Transformer aggregator (CLS + pos)  →  (B, 384)  =  global_vec
        │
        └──► Local branch
                extract_tubes:
                    spatial unfold (grid_size × grid_size cells)
                    temporal sliding windows (tube_length, temporal_stride)
                    →  (B, N_tubes, 3, T_t, h, w)
                LocalTubeEncoder per tube:
                    tubelet Conv3d (t_kernel × h_patch × w_patch)
                    → tokens + CLS → self-attention
                    →  (B, N_tubes, 384)
                tube_validity_mask  →  (B, N_tubes)
                                 │
                                 ▼
        FusionModule (×2 blocks):
            self-attn over local tubes
            cross-attn: global=Q, locals=KV
            →  (B, 384)
                                 │
                                 ▼
        MLP head (Linear→GELU→Linear)  →  logit  (B,)
```

### Component summary

| Module | File | Role |
|--------|------|------|
| `GlobalBranch` | `src/.../global_branch.py` | Frozen DINOv2 ViT-S/14 (last block finetuned) + 2-layer Transformer aggregator. |
| `LocalBranch` | `src/.../local_branch.py` | `extract_tubes` + tubelet Conv3d + self-attn video transformer per tube. |
| `FusionModule` | `src/.../fusion.py` | Self-attn over local tokens + cross-attn (global=Q, locals=KV). |
| `TubeMultiscaleClassifier` | `src/.../classifier.py` | Glues the three modules + MLP classification head. |
| `LitTubeMultiscaleClassifier` | `src/.../lit_module.py` | Lightning wrapper: BCE, AdamW two-group, cosine warmup, F1 callbacks. |

### Default geometry

Picked from the resolution sweep (see Results): a **2×2 spatial grid of
112×112 cells** with **4-frame temporal windows at stride 2** — yielding
7 × 4 = **28 tubes per sequence**.

All knobs are exposed in `params.yaml` under `local_branch.*`.

## Data

Reuses the upstream pipeline from `bbox-tube-motion-fusion`:

- `pyro-dataset v2.2.0` (sequential train/val) imported via DVC, truncated to
  16 frames per sequence.
- Greedy IoU tube matching + 1.5× context bbox crops, resized to 224×224
  PNGs at `data/05_model_input/{train,val}/`.

```bash
uv sync
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/train \
    -o data/01_raw/datasets_full/train --rev v2.2.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/val \
    -o data/01_raw/datasets_full/val --rev v2.2.0
uv run dvc pull
```

Train / val tube counts after IoU-based filtering: 2 831 / 280.

## Results

Resolution sweep (single-seed, same training schedule and 5-epoch early-stop
patience on `val/f1`):

| variant            | grid / tube_len / stride | tubes/seq | train F1 | val F1 | val acc | val prec | val rec | val PR-AUC |
|--------------------|--------------------------|----------:|---------:|-------:|--------:|---------:|--------:|-----------:|
| **default** (2×2)  | 2×2 / 4 / 2              |        28 |   0.9810 | **0.9783** | 0.9786 | 0.9574 | **1.0000** | 0.9936 |
| spatial_4x4        | 4×4 / 4 / 2              |       112 |   0.9344 | 0.9673 |  0.9679 |   0.9500 |  0.9852 |     0.9874 |
| spatial_8x8        | 8×8 / 4 / 2              |       448 |   0.9701 | 0.9745 |  0.9750 |   0.9571 |  0.9926 |     0.9942 |
| temporal_stride1   | 2×2 / 4 / 1              |        52 |   0.9769 | 0.9781 |  0.9786 |   0.9640 |  0.9926 | **0.9954** |
| temporal_len8      | 2×2 / 8 / 4              |        12 |   0.9760 | 0.9744 |  0.9750 |   0.9638 |  0.9852 |     0.9939 |

> Note: sweep variants `temporal_stride1` and `temporal_len8` use the new
> 2×2 spatial default; their tube counts are listed accordingly.

**Confusion matrix (val, 280 sequences)** of the default 2×2 variant:

| | predicted fp | predicted smoke |
|---:|---:|---:|
| **fp**    | 139 (TN) |   6 (FP) |
| **smoke** |   0 (FN) | 135 (TP) |

Compared to the **bbox-tube-motion-fusion** baseline:

| model | val F1 | val acc | val prec | val rec | val PR-AUC | extra inputs |
|---|---:|---:|---:|---:|---:|---|
| bbox-tube-motion-fusion (`vit_dinov2_motion_frame_diff`) | 0.9776 | — | — | — | — | precomputed motion images |
| **tube-multiscale-fusion (default 2×2)** | **0.9783** | 0.9786 | 0.9574 | **1.0000** | 0.9936 | RGB only |

Same backbone (DINOv2 ViT-S/14, last block fine-tuned), same training schedule
class, with **no motion-feature precomputation** — the local tube transformer
learns the equivalent signal from raw pixels.

### Benchmarking

Standardized Pyronear R&D metrics (see [GUIDELINES.md](../../GUIDELINES.md))
for the default 2×2 model on the val split (280 tubes), computed by
`scripts/benchmark.py`. Hardware: NVIDIA RTX 4090 (GPU) / 24-thread CPU.

| Metric | Value |
|--------|-------|
| **Recall @ FPR=1%** | 0.874 |
| **Recall @ FPR=5%** | 1.000 |
| **Recall @ FPR=10%** | 1.000 |
| **Time-to-detection** (median) | 2 frames ≈ 30 s after tube start |
| **Time-to-detection** (mean) | 2.07 frames ≈ 32 s; 135/135 positives eventually fire |
| **Inference latency (GPU)** | 10.7 ms/sequence → 0.67 ms/frame |
| **Inference latency (CPU)** | 290 ms/sequence → 18.2 ms/frame |
| **Model size** | 36.4 M params (16.6 M trainable, 19.9 M frozen DINOv2) |
| **FLOPs** | 226.5 GFLOPs/sequence → 14.2 GFLOPs/frame |

Notes:

- **Recall @ FPR** is the tube-level ROC operating point: at a 5 % false-alarm
  rate every smoke tube in the val set is recovered. The 1 % point (0.874) is
  the conservative-threshold regime relevant for low-nuisance deployment.
- **Time-to-detection** is measured by feeding the model increasing frame
  prefixes (k = 2…16, later frames masked — the same variable-length contract
  used in training) and recording the first prefix whose probability crosses
  0.5. Reported as `(k_fire − 1) × Δt`, with Δt = 30 s the median inter-frame
  interval in the val sequences. Median detection at the 2nd frame means smoke
  is typically flagged within one frame interval of the tube's first detection.
- **FLOPs** is dominated by the 16 per-frame DINOv2 ViT-S/14 forward passes;
  the local tube transformer and fusion add a small fraction on top.

Reproduce with:

```bash
uv run dvc repro benchmark_dinov2_multiscale
```

### Ablation: remove the temporal module (global branch)

To isolate what the **global DINOv2 sequence branch** (the "temporal module")
contributes, we train a faithful ablation: the **local branch and the fusion
module are the exact same modules with the same hyperparameters** as the full
model, and the *only* change is that the global branch is removed. Because the
fusion's cross-attention needs a query (the global context vector in the full
model), that single input is replaced by a **learnable query token** — a
constant, data-independent parameter. Everything else (local tube branch,
fusion self-attention + cross-attention + FFN, MLP head) is byte-for-byte
identical. Same tube geometry (2×2 / len 4 / stride 2 → 28 tubes/seq), schedule,
and data.

Removing the global branch removes the 16 per-frame DINOv2 forward passes that
dominate the full model's compute, so the ablation is strictly *smaller and
cheaper* — exactly the point of an ablation.

| | **Full model** (global + local + fusion) | **Ablation** (no temporal module) |
|---|---:|---:|
| val F1 | **0.9783** | 0.8716 |
| val accuracy | **0.9786** | 0.8643 |
| val precision | **0.9574** | 0.8012 |
| val recall | **1.0000** | 0.9556 |
| val PR-AUC | **0.9936** | 0.9019 |
| val FP / FN (of 280) | **6 / 0** | 32 / 6 |
| Recall @ FPR=1% | **0.874** | 0.193 |
| Recall @ FPR=5% | **1.000** | 0.563 |
| Recall @ FPR=10% | **1.000** | 0.748 |
| Params (trainable) | 36.4 M (16.6 M) | **11.3 M** (11.3 M) |
| GFLOPs / sequence | 226.5 | **30.5** |
| GPU latency (ms/seq) | 10.7 | **2.1** |

**Takeaway.** Removing the temporal module costs ~11 F1 points, and the damage
is concentrated exactly where the design predicts: **precision and the
low-false-alarm regime**, not recall. False positives jump 5× (6 → 32),
precision drops 0.957 → 0.801, and recall at a strict 1 % FPR collapses
0.874 → 0.193 — while recall stays comparatively high (1.000 → 0.956). In other
words, the local tube branch alone still *finds* smoke, but without the global
context it can no longer reliably *reject* slow-moving look-alikes (cloud, fog,
haze): local turbulence alone is an ambiguous cue, and it takes the long-range
sequence context to confirm whether that motion is a growing plume or benign
weather. The ablation is ~7× cheaper (30.5 vs 226.5 GFLOPs) precisely because it
drops the DINOv2 sequence branch — but that branch is what buys the
deployment-critical low-FPR operating point.

Reproduce with:

```bash
uv run dvc repro train_ablation_no_temporal evaluate_ablation_no_temporal
```

## How to Reproduce

```bash
uv sync
# Import data (see "Data" above)
uv run dvc repro
```

Run just the default training and packaging:

```bash
uv run dvc repro train_dinov2_multiscale
uv run dvc repro evaluate_dinov2_multiscale
uv run dvc repro package_dinov2_multiscale
```

Re-run only the resolution sweep:

```bash
uv run dvc repro train_dinov2_multiscale_sweep evaluate_dinov2_multiscale_sweep
```

## Packaged model

`scripts/package_model.py` produces a self-contained
`data/07_model_output/dinov2_multiscale/model_package.zip` (~245 MB) with:

- `checkpoint.pt` — Lightning checkpoint
- `params.yaml` — exactly the `global_branch`, `local_branch`, `fusion`, and
  training-variant blocks used to train this checkpoint
- `manifest.json` — package version, SHA-256, params key
- `README.md` — loading instructions and inference contract

To load the packaged model:

```python
import zipfile, tempfile
from pathlib import Path
from tube_multiscale_fusion.lit_module import LitTubeMultiscaleClassifier

with zipfile.ZipFile("model_package.zip") as zf, tempfile.TemporaryDirectory() as td:
    zf.extractall(td)
    lit = LitTubeMultiscaleClassifier.load_from_checkpoint(
        str(Path(td) / "checkpoint.pt"), pretrained=False
    )
    lit.eval()
```

## Tests

```bash
make install
make test                            # all tests
uv run pytest tests/ -m "not slow"   # skip the heavy DINOv2 download tests
```

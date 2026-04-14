# Basic Temporal Model — Design

**Date:** 2026-04-14
**Status:** Approved, ready for implementation plan
**Scope:** `experiments/temporal-models/smokeynet-adapted/`

## Goal

Build the smallest credible vision-based temporal smoke classifier on top
of the existing tube data. One tube → one binary prediction (`smoke` vs
`fp`). Two architectures share one training script:

- **A — `mean_pool`**: frozen pretrained CNN per frame, masked mean-pool
  over time, MLP head. Sanity baseline; confirms the visual signal
  exists in the cropped patches.
- **B — `gru`**: frozen pretrained CNN per frame, 1-layer GRU over
  time, MLP head. The actual "basic temporal model".

Both archs share dataset, backbone wrapper, and Lightning training code;
only the temporal head differs. The backbone is a `timm` model selected
by string from per-stage params (`resnet18` by default, swappable to any
timm model that supports `num_classes=0, global_pool="avg"`).

## Pipeline

New DVC stages added after `build_tubes`:

```
truncate → build_tubes → build_model_input → train_mean_pool → evaluate_mean_pool_{train,val}
                                            ↘ train_gru       → evaluate_gru_{train,val}
```

`render_tubes` remains a sibling reporting stage on `build_tubes` and is
unchanged. Both training stages consume the same `data/05_model_input/`
output and run independently; one `dvc repro` produces both checkpoints
and all four evaluate reports.

## Stage 1 — `build_model_input`

### Inputs

```
data/03_primary/tubes/{train,val}/<sequence_id>.json
data/01_raw/datasets/{train,val}/{wildfire,fp}/<sequence_id>/images/*.jpg
```

### Outputs

```
data/05_model_input/{train,val}/
  <sequence_id>/
    frame_00.png ... frame_NN.png   # 224×224 uint8 RGB cropped patches
    meta.json
  _index.json
```

`meta.json` per tube:

```json
{
  "sequence_id": "...",
  "split": "train",
  "label": "smoke",            // or "fp"
  "label_int": 1,              // 0 = fp, 1 = smoke
  "num_frames": 20,
  "context_factor": 1.5,
  "patch_size": 224,
  "frames": [
    {
      "frame_idx": 0,
      "frame_id": "...",
      "is_gap": false,
      "orig_bbox": [cx, cy, w, h],
      "crop_bbox_pixels": [x0, y0, x1, y1],   // pre-resize crop in source image
      "filename": "frame_00.png"
    },
    ...
  ]
}
```

`_index.json` per split: flat list `[{"sequence_id": ..., "label_int": ..., "num_frames": ...}, ...]`
for fast Dataset construction without scanning subfolders.

### Crop logic

For each tube entry:

1. Take `bbox = [cx, cy, w, h]` in normalized coordinates.
2. Expand to `bbox' = [cx, cy, w * cf, h * cf]` with `cf = 1.5`.
3. Convert to pixel coordinates using the source image dimensions.
4. Square the box by enlarging the smaller side to match the larger
   (centered), then clip to image bounds. If clipping makes the box
   non-square, pad the resulting crop with zeros on the deficient side
   to recover a square before resize. This guarantees square output
   without distortion when bboxes hit the image edge.
5. Resize to `224×224` with bilinear interpolation.
6. Save as PNG (uint8, RGB).

Gap frames (`is_gap: true`) are cropped from the raw frame at the
interpolated bbox — same code path. The tube builder already bounds
gaps via `max_misses=2` and `min_detected_entries=2`.

### Parallelism

`ProcessPoolExecutor`, one worker per tube (or batched). Mirrors the
existing `render_tubes` parallelization.

### Params (additions to `params.yaml`)

```yaml
model_input:
  context_factor: 1.5
  patch_size: 224
  image_format: png
```

### Code

- `src/smokeynet_adapted/model_input.py` — pure functions for crop +
  save logic, importable and unit-testable.
- `scripts/build_model_input.py` — CLI orchestrator using DVC params.

## Stage 2 — `train_mean_pool` and `train_gru`

Two independent DVC stages, one per architecture. Both invoke the same
`scripts/train.py` with `--arch {mean_pool|gru}` and `--params-key
{train_mean_pool|train_gru}`.

### Dataset (`src/smokeynet_adapted/dataset.py`, full rewrite)

Constructor reads `_index.json`. `__getitem__(i)` returns:

```python
{
  "patches": Tensor[T, 3, 224, 224],   # float32, ImageNet-normalized
  "mask":    Tensor[T],                 # bool, True = real frame
  "label":   Tensor[],                  # float32 0.0 or 1.0
  "sequence_id": str,
}
```

- Loads PNG frames from disk via `torchvision.io.read_image` (uint8 →
  float32 / 255, then `Normalize(mean=ImageNet, std=ImageNet)`).
- Pads to `T = max_frames = 20` left-aligned with zero patches; mask
  marks real frames.
- No augmentation.
- `num_workers` controlled by params.

### Model (`src/smokeynet_adapted/model.py`, full rewrite)

```
class TemporalSmokeClassifier(nn.Module):
    backbone: timm.create_model(cfg.backbone, pretrained=True,
                                num_classes=0, global_pool="avg")
              → outputs (B, feat_dim) per frame; feat_dim is read from
              backbone.num_features (auto-detects per backbone).
              Frozen: requires_grad=False, .eval() mode permanently.
    head:     "mean_pool" or "gru"
              - mean_pool: MLP(feat_dim → hidden → 1), masked mean over T
              - gru:       GRU(feat_dim → hidden, num_layers, batch_first=True,
                               bidirectional) with packed sequences, then
                           MLP(hidden*(2 if bidir else 1) → 1)
    forward(patches, mask) -> logits[B]
```

Implements `pyrocore.TemporalModel` interface for downstream use.

`timm` is added to `pyproject.toml` dependencies (`timm>=1.0`).

### Lightning Module (`src/smokeynet_adapted/training.py`, full rewrite)

- Loss: `BCEWithLogitsLoss` (no `pos_weight`, data is balanced).
- Optimizer: `AdamW(head_params, lr, weight_decay)`. Backbone params
  excluded.
- No scheduler.
- Logs per-step train loss; per-epoch val loss + accuracy/precision/
  recall/F1 at threshold 0.5.

### Training script (`scripts/train.py`)

CLI args: `--arch {mean_pool|gru}`, `--train-dir`, `--val-dir`,
`--output-dir`, `--params-path`, `--params-key`. Loads only the named
params section from `params.yaml`.

- `pl.Trainer(max_epochs, callbacks=[EarlyStopping(monitor="val/f1", mode="max", patience), ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=1, filename="best")])`.
- TensorBoard + CSV logger to `<output-dir>/`.
- Fixed seed via `pl.seed_everything(seed)`.

### Outputs

```
data/06_models/mean_pool/
  best_checkpoint.pt
  csv_logs/
  tb_logs/

data/06_models/gru/
  best_checkpoint.pt
  csv_logs/
  tb_logs/
```

### Params (additions to `params.yaml`, replaces existing `train:`)

Each training stage has its OWN flat top-level section. Duplication
between sections is intentional — DVC tracks per-section changes
precisely, and it keeps each stage self-contained.

```yaml
train_mean_pool:
  arch: mean_pool
  backbone: resnet18      # any timm model id supported by
                           # create_model(..., num_classes=0, global_pool="avg")
  hidden_dim: 128
  max_frames: 20
  batch_size: 32
  num_workers: 4
  learning_rate: 0.001
  weight_decay: 0.01
  max_epochs: 30
  early_stop_patience: 5
  seed: 42

train_gru:
  arch: gru
  backbone: resnet18
  hidden_dim: 128
  num_layers: 1
  bidirectional: false
  max_frames: 20
  batch_size: 32
  num_workers: 4
  learning_rate: 0.001
  weight_decay: 0.01
  max_epochs: 30
  early_stop_patience: 5
  seed: 42
```

## Stage 3 — `evaluate_mean_pool_{train,val}` and `evaluate_gru_{train,val}`

Four `evaluate` stages, one per (arch × split). Each invokes
`scripts/evaluate.py` with the matching checkpoint and split.

### Inputs

`data/06_models/{arch}/best_checkpoint.pt` + `data/05_model_input/{split}/`

### Outputs

```
data/08_reporting/{split}/{arch}/
  metrics.json   # accuracy, precision, recall, f1, pr_auc, roc_auc,
                 # confusion_matrix at threshold 0.5
  pr_curve.png
  roc_curve.png
```

DVC `metrics:` and `plots:` declarations on `metrics.json` and the
PNGs. `dvc metrics show` then displays a side-by-side table of all four
runs without manual orchestration.

## DVC Stage Definitions

```yaml
build_model_input:
  foreach: [train, val]
  do:
    cmd: >-
      uv run python scripts/build_model_input.py
      --tubes-dir data/03_primary/tubes/${item}
      --raw-dir data/01_raw/datasets/${item}
      --output-dir data/05_model_input/${item}
      --context-factor ${model_input.context_factor}
      --patch-size ${model_input.patch_size}
    deps:
      - scripts/build_model_input.py
      - src/smokeynet_adapted/model_input.py
      - data/03_primary/tubes/${item}
      - data/01_raw/datasets/${item}
    params: [model_input]
    outs:
      - data/05_model_input/${item}

train_mean_pool:
  cmd: >-
    uv run python scripts/train.py
    --arch mean_pool
    --train-dir data/05_model_input/train
    --val-dir data/05_model_input/val
    --output-dir data/06_models/mean_pool
    --params-path params.yaml
    --params-key train_mean_pool
  deps:
    - scripts/train.py
    - src/smokeynet_adapted/dataset.py
    - src/smokeynet_adapted/model.py
    - src/smokeynet_adapted/training.py
    - data/05_model_input/train
    - data/05_model_input/val
  params: [train_mean_pool]
  outs:
    - data/06_models/mean_pool/best_checkpoint.pt
  plots:
    - data/06_models/mean_pool/csv_logs/

train_gru:
  cmd: >-
    uv run python scripts/train.py
    --arch gru
    --train-dir data/05_model_input/train
    --val-dir data/05_model_input/val
    --output-dir data/06_models/gru
    --params-path params.yaml
    --params-key train_gru
  deps:
    - scripts/train.py
    - src/smokeynet_adapted/dataset.py
    - src/smokeynet_adapted/model.py
    - src/smokeynet_adapted/training.py
    - data/05_model_input/train
    - data/05_model_input/val
  params: [train_gru]
  outs:
    - data/06_models/gru/best_checkpoint.pt
  plots:
    - data/06_models/gru/csv_logs/

evaluate_mean_pool:
  foreach: [train, val]
  do:
    cmd: >-
      uv run python scripts/evaluate.py
      --arch mean_pool
      --data-dir data/05_model_input/${item}
      --checkpoint data/06_models/mean_pool/best_checkpoint.pt
      --output-dir data/08_reporting/${item}/mean_pool
      --params-path params.yaml
      --params-key train_mean_pool
    deps:
      - scripts/evaluate.py
      - src/smokeynet_adapted/model.py
      - src/smokeynet_adapted/dataset.py
      - data/06_models/mean_pool/best_checkpoint.pt
      - data/05_model_input/${item}
    params: [train_mean_pool]
    metrics:
      - data/08_reporting/${item}/mean_pool/metrics.json:
          cache: false
    plots:
      - data/08_reporting/${item}/mean_pool/pr_curve.png
      - data/08_reporting/${item}/mean_pool/roc_curve.png

evaluate_gru:
  foreach: [train, val]
  do:
    cmd: >-
      uv run python scripts/evaluate.py
      --arch gru
      --data-dir data/05_model_input/${item}
      --checkpoint data/06_models/gru/best_checkpoint.pt
      --output-dir data/08_reporting/${item}/gru
      --params-path params.yaml
      --params-key train_gru
    deps:
      - scripts/evaluate.py
      - src/smokeynet_adapted/model.py
      - src/smokeynet_adapted/dataset.py
      - data/06_models/gru/best_checkpoint.pt
      - data/05_model_input/${item}
    params: [train_gru]
    metrics:
      - data/08_reporting/${item}/gru/metrics.json:
          cache: false
    plots:
      - data/08_reporting/${item}/gru/pr_curve.png
      - data/08_reporting/${item}/gru/roc_curve.png
```

## File Layout (new / rewritten)

```
scripts/
  build_model_input.py   (new)
  train.py               (new)
  evaluate.py            (new)

src/smokeynet_adapted/
  model_input.py         (new — crop/save logic)
  dataset.py             (rewrite)
  model.py               (rewrite)
  training.py            (rewrite — Lightning module)

tests/
  test_model_input.py    (new — crop math, gap handling, edge clipping)
  test_dataset.py        (new — padding, mask, normalization)
  test_model.py          (new — forward shape, mask respects, both archs)
```

Existing modules `backbone.py`, `detector.py`, `heads.py`, `net.py`,
`spatial_attention.py`, `temporal_fusion.py`, `package.py` are NOT
touched in this iteration. They were scaffolded for the heavier
SmokeyNet plan and remain available for later iterations.

## Out of Scope (deliberately deferred)

- Data augmentation
- Backbone fine-tuning (frozen for v1)
- Multi-tube fusion (one tube = one prediction)
- Detection-level / per-frame loss
- Attention or transformer temporal models
- ROI pooling on backbone feature maps (we re-crop pixels)
- Model packaging (`scripts/package.py`)
- Inference-time integration with the production server

## Success Criteria

1. `dvc repro` runs end-to-end without manual steps and trains BOTH
   `train_mean_pool` and `train_gru` stages plus all four evaluate
   stages.
2. Both `mean_pool` and `gru` produce metrics.json files for train and
   val splits.
3. `gru` achieves val F1 strictly greater than `mean_pool` (validates
   that temporal modeling adds signal); both must clearly beat random
   (val F1 > 0.6).
4. PNG patches in `data/05_model_input/` are visually inspectable in a
   file browser and the smoke is recognizably centered in positive
   tubes.
5. Unit tests for crop math, dataset padding/masking, and model
   forward/mask behavior all pass.

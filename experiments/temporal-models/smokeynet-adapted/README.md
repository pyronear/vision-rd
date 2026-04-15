# smokeynet-adapted

Temporal smoke classifier for Pyronear camera sequences. Learns a binary sequence-level label (smoke / no smoke) over short tubes of YOLO-detected regions.

## Architecture

```
raw sequence  ->  truncate  ->  build tubes  ->  224x224 patches  ->  timm backbone  ->  temporal head  ->  sequence logit
                  (max 20f)    (greedy IoU      (per tube entry,     (frozen or        (mean-pool or
                                matching)       context-expanded)     last-N fine-     GRU)
                                                                      tuned)
```

- **Tubes**: chains of YOLO detections linked across frames by greedy IoU matching (`src/smokeynet_adapted/tubes.py`). YOLO bboxes are pre-computed and shipped with the dataset; we do not run YOLO in this experiment.
- **Patches**: each tube entry is cropped to a context-expanded 224x224 RGB patch (`src/smokeynet_adapted/model_input.py`).
- **Backbone**: a `timm` model (`resnet18` or `convnext_tiny`) applied per-frame, either frozen or with the last N blocks fine-tuned.
- **Temporal head**: `mean_pool` (average per-frame features) or `gru` (single-layer GRU over the tube).
- **Augmentations**: per-tube-consistent spatial, photometric, and temporal transforms (`src/smokeynet_adapted/augment.py`).

## Quick start

```bash
make install   # uv sync + nbstripout
make lint      # ruff check
make test      # pytest
```

## DVC pipeline

```bash
uv run dvc repro                 # full pipeline
uv run dvc repro train_gru       # up to a specific train stage
uv run dvc metrics show          # per-variant metrics
uv run dvc exp run -S train_gru.seed=100
```

Stages:

| Stage | Purpose |
|-------|---------|
| `truncate` (foreach train/val) | Cap sequences at `truncate.max_frames`. |
| `build_tubes` (foreach train/val) | Link detections into tubes via greedy IoU. |
| `build_model_input` (foreach train/val) | Crop 224x224 patches per tube entry. |
| `render_tubes` (foreach train/val) | Sanity-check tube visualizations. |
| `train_<variant>` | Train one variant (Lightning + CSV logger + training plots). |
| `evaluate_<variant>` (foreach train/val) | Per-sequence predictions + metrics. |
| `compare_variants` | Aggregate a comparison markdown across all variants. |

## Variants

All variants share the `truncate -> build_tubes -> build_model_input` inputs; they differ only in backbone, temporal head, seed, and whether the backbone is fine-tuned.

| Variant | Backbone | Head | Seed | Fine-tune |
|---------|----------|------|------|-----------|
| `train_mean_pool` | resnet18 | mean-pool | 42 | no |
| `train_gru` | resnet18 | GRU | 42 | no |
| `train_gru_seed43` | resnet18 | GRU | 43 | no |
| `train_gru_seed44` | resnet18 | GRU | 44 | no |
| `train_gru_convnext` | convnext_tiny | GRU | 42 | no |
| `train_gru_finetune` | resnet18 | GRU | 42 | last 1 block |
| `train_gru_convnext_finetune` | convnext_tiny | GRU | 42 | last 1 block |

## Reproducibility

Training is seeded end-to-end: same `seed` + same hardware + same `num_workers` → bitwise-identical final weights and logged metrics. The seed is configured per variant as `train_<variant>.seed` in `params.yaml` and can be overridden ad hoc via `uv run dvc exp run -S train_gru.seed=100`.

Mechanism (in `scripts/train.py`): `L.seed_everything(cfg["seed"], workers=True)` seeds torch / numpy / random / `PYTHONHASHSEED` and causes Lightning to inject a deterministic `worker_init_fn` into every DataLoader; `Trainer(deterministic=True)` sets `torch.use_deterministic_algorithms(True)`, disables cuDNN benchmark mode, and sets `CUBLAS_WORKSPACE_CONFIG=":4096:8"`.

Caveats: GPU reproducibility holds on the same GPU model + driver + CUDA version; bitwise equality across different GPUs is not guaranteed. The executable spec is `tests/test_reproducibility.py`, which fits two short runs on CPU with the same seed and asserts every `state_dict` tensor is identical.

## Key params

See `params.yaml`. Highlights:

- `truncate.max_frames`, `tubes.iou_threshold`, `tubes.max_misses`, `build_tubes.min_tube_length` — shape the inputs.
- `model_input.context_factor`, `model_input.patch_size` — patch geometry.
- `train_<variant>.*` — per-variant hyperparameters (lr, batch size, epochs, early stopping, seed, backbone/head config).
- `augment.*` — spatial / photometric / temporal augmentation toggles.

## Notebooks

- `notebooks/02-visualize-built-tubes.ipynb` — inspect tubes produced by the `build_tubes` stage.
- `notebooks/03-visualize-augment-transforms.ipynb` — preview the augmentation pipeline on real tubes.

## Layout

Kedro-style data layers under `data/` (`01_raw`, `03_primary`, `05_model_input`, `06_models`, `07_model_output`, `08_reporting`); source under `src/smokeynet_adapted/`; CLI entry points under `scripts/`; design docs under `docs/specs/` and `docs/plans/`.

# data-quality/sequential

Use a `pyrocore.TemporalModel` oracle to surface probably-mis-labeled
sequences in the pyro-dataset sequential splits (`train`/`val`/`test`).
For each registered model, the pipeline runs inference on every sequence
and flags disagreements against folder-based ground truth
(`wildfire/` = positive, `fp/` = negative) as two FiftyOne review sets:

- **FP set**: model predicted positive, sequence is under `fp/`.
- **FN set**: model predicted negative, sequence is under `wildfire/`.

Reviewers browse the FiftyOne datasets and decide whether each flag is
a real label error or a true model mistake.

## Design

See [`../../../docs/specs/2026-04-23-sequential-label-audit-design.md`](../../../docs/specs/2026-04-23-sequential-label-audit-design.md).

## Pipeline

```
predict  →  build_review_sets  →  build_fiftyone
 (split × model)     (split × model)     (split × model)
```

Adding a new model variant: see "Adding another model" below.

## How to reproduce

```bash
cd experiments/data-quality/sequential
make install

# Fetch the imported dataset + model zip via your usual `dvc pull` workflow.

# Full pipeline (train + val + test × all models):
uv run dvc repro
```

Reports land in `data/08_reporting/<model>/<split>/`; FiftyOne datasets
named `dq-seq_<model>_<split>_fp` and `dq-seq_<model>_<split>_fn` are
created in the local FiftyOne mongo store.

Browse the review sets:

```bash
make fiftyone
```

Opens the FiftyOne app (default: http://localhost:5151). Use the
sidebar to switch between `dq-seq_<model>_<split>_{fp,fn}` datasets.
To load a specific dataset directly:

```bash
uv run --group explore fiftyone app launch dq-seq_bbox-tube-temporal-vit-dinov2-finetune_val_fp
```

## Data imports

`train`/`val`/`test` are imported from [`pyro-dataset`](https://github.com/pyronear/pyro-dataset)
at tag `v3.0.0`:

```bash
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/train \
    -o data/01_raw/datasets/train --rev v3.0.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/val \
    -o data/01_raw/datasets/val --rev v3.0.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_test \
    -o data/01_raw/datasets/test --rev v3.0.0
```

Model packages are imported from this repo (the `bbox-tube-temporal`
experiment must have `dvc push`ed the chosen variant at the pinned ref
first):

```bash
uv run dvc import git@github.com:pyronear/vision-rd.git \
    experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip \
    -o data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip \
    --rev main
```

## Adding another model

1. `dvc import` its packaged `.zip` into `data/01_raw/models/<model-name>.zip`
   (the filename must equal the model-name used in pipeline paths).
2. If the model's registry key isn't already in
   `src/data_quality_sequential/registry.py::MODEL_REGISTRY`, add it (one line
   — see the leaderboard's `temporal_model_leaderboard.registry` for the
   shape). The class must expose `classmethod from_package(path)`.
3. Append three new stage blocks to `dvc.yaml` (copy the existing triple,
   replace the model-name everywhere). This mirrors the leaderboard's
   per-model stage-block convention.
4. `uv run dvc repro` — only the new model's stages run.

## Caveats

- Frames are sorted by filename in `dataset.iter_sequences`. This gives
  the correct temporal order only because pyro-dataset frames end in a
  zero-padded `YYYY-MM-DDTHH-MM-SS.jpg` stamp. The glob is also
  case-sensitive and `.jpg`-only. If a future dataset revision changes
  the filename convention or extension, switch to timestamp-aware sort
  (see `pyrocore.model._try_parse_timestamp`) before trusting
  `trigger_frame_index` values.

## Layout

```
data/
  01_raw/
    datasets/{train,val,test}.dvc       # dvc-imported
    models/<model>.zip.dvc              # dvc-imported
  07_model_output/<model>/<split>/
    predictions.json
  08_reporting/<model>/<split>/
    fp_sequences.csv
    fn_sequences.csv
    summary.json
    review_manifest.json
  fiftyone/<model>/<split>/
    datasets.json                       # sentinel; actual datasets in mongo
```

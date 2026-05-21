# Tube Builder Lab

A standalone Streamlit lab to iterate on the **bbox-tube linking algorithm**.
For a set of real failure-case sequences it shows, side by side, the tubes the
**current** builder produces vs. the tubes an editable **candidate** builder
produces — so you can fix over-fragmentation and confirm it by eye.

Run every command from this directory:

```bash
cd experiments/temporal-models/tube-builder-lab
```

## Run the lab (the common case)

The sequences, per-frame detections, and pipeline config are all DVC-tracked, so
you just pull them and launch — **no model and no regeneration needed**:

```bash
make install      # uv sync + nbstripout
dvc pull          # fetch data/03_primary/sequences + data/05_model_input/*
make app          # launch at http://localhost:8501
```

Iterate by editing `src/tube_builder_lab/candidate.py` and saving — Streamlit
re-runs and the lab reloads your candidate automatically (hit **R** to force a
rerun). The working-set tables highlight every sequence whose tube count changes.

## Regenerating the data (only when you change inputs)

`bootstrap`/`import` and `cache` are **one-time generation steps** — you do NOT
need them after a `dvc pull`. Run them only to create the data the first time, or
to refresh it after editing `working_set.yaml`, the detection params, or the
model:

```bash
# 1. get the sequence frames (pick ONE):
make bootstrap                       # copy from the local temporal-model-explorer store (no creds)
# or: uv run python scripts/import_sequences.py   # fetch by id from the platform (needs creds)

# 2. run YOLO once and cache detections (needs a model package, see below):
make cache                           # -> data/05_model_input/detections/ + pipeline_config.yaml

# 3. publish so others can `dvc pull`:
uv run dvc add data/03_primary/sequences   # refresh the sequences pointer
uv run dvc commit cache_detections         # track the regenerated detections
uv run dvc push
```

`make cache` needs a model package at `data/06_models/<name>/model.zip` (set by
`model_name` in `params.yaml`); only its bundled YOLO + detection config are used.
The model is a **local input**, not DVC-tracked — it's only needed to regenerate
detections, never to run the app.

## Pipeline

```
platform / explorer ──import|bootstrap──> data/03_primary/sequences/<key>/   [DVC]
model.zip           ──cache_detections──> data/05_model_input/detections/    [DVC]
                                          + pipeline_config.yaml             [DVC]
detections + candidate.py ──> app.py (current vs candidate, side by side)
```

## Common commands

```bash
make install     # uv sync
make bootstrap   # copy working-set sequences from the local explorer store
make cache       # run YOLO + cache detections for the working set
make app         # launch the lab
make test        # pytest
make lint        # ruff check
make format      # ruff format
```

## Working set

`working_set.yaml` holds the curated sequences: `targets` (collected
over-fragmentation cases) and `control` (random sequences watched for
regressions; reviewed ones are noted `verified ok`). `tests/test_regression.py`
pins the candidate's tube counts on these (skipped when the DVC data is absent).

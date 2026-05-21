# Temporal Model Explorer

A local Streamlit tool to run and inspect second-stage **temporal smoke models**
(keep / discard) over platform sequences. For each sequence it shows the frames
with YOLO bboxes overlaid, the extracted smoke tubes, and the model's decision —
so you can eyeball where the model keeps real smoke and where it filters false
positives.

This is a self-contained `uv` project; run every command from this directory:

```bash
cd experiments/temporal-models/temporal-model-explorer
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages the Python 3.11+ env and deps)
- Access to the DVC S3 remote (to fetch the tracked sequences/outputs)
- A GPU is optional — scoring runs much faster on CUDA but works on CPU
- Platform API credentials are only needed if you import fresh sequences

## Quick start

```bash
make install          # uv sync
uv run dvc pull       # fetch sequences (+ model outputs) from S3
make app              # launch the viewer at http://localhost:8501
```

If the model outputs aren't available yet, or you changed `params.yaml`,
regenerate them with the pipeline before launching the app:

```bash
uv run dvc repro      # prepare_models -> run_models
```

## How it fits together

```
platform API ──import_platform──> data/03_primary/sequences/   (frames + bbox meta)
params.yaml  ──prepare_models───> data/06_models/<name>/        (model.zip)
sequences + models ──run_models─> data/07_model_output/         (results.parquet + details/)
                                          │
                                          └──> Streamlit app (read-only viewer)
```

The app only **reads** `data/07_model_output/` and `data/03_primary/sequences/`;
it never runs models or calls the API itself.

## Importing fresh sequences (optional)

The credentials determine which organization's data you pull, so set them for the
org you want:

```bash
export PLATFORM_API_ENDPOINT=https://...
export PLATFORM_LOGIN=...
export PLATFORM_PASSWORD=...
# optional admin creds, only used to resolve organization names:
export PLATFORM_ADMIN_LOGIN=...
export PLATFORM_ADMIN_PASSWORD=...

uv run python scripts/import_platform.py --date-from 2026-05-19 --date-to 2026-05-19
uv run dvc repro run_models     # score the newly imported sequences
```

The date range is required on the command line; the detection limit and camera
filter come from `params.yaml` (`platform.*`).

## Common commands

```bash
make install          # uv sync
make app              # launch the Streamlit viewer
make test             # pytest tests/ -v
make lint             # ruff check
make format           # ruff format
make help             # list targets
```

## CLIs

Each pipeline stage is also a standalone script:

| Script | Purpose | Key flags |
| --- | --- | --- |
| `scripts/import_platform.py` | Import platform sequences into the store | `--date-from`, `--date-to` (required), `--out` |
| `scripts/prepare_models.py` | Copy the `model.zip`s named in `params.yaml` into `data/06_models/` | `--out`, `--params` |
| `scripts/run_models.py` | Score every sequence and write `results.parquet` + `details/` | `--device {auto,cpu,cuda}`, `--store`, `--models-dir`, `--out` |

`run_models.py` defaults to `--device auto` (CUDA when available, else CPU).

## Configuration (`params.yaml`)

- `platform.*` — import detection limit and camera filter
- `label_mapping.*` — which annotation values count as smoke vs. false positive
- `models` — `name -> source model.zip path`, copied in by `prepare_models`

## Data layout

Kedro-style layers, tracked by DVC (data lives in S3, not git):

- `data/03_primary/sequences/<org>/` — imported sequences (frames + bbox metadata)
- `data/06_models/<name>/` — the model artifact used for scoring
- `data/07_model_output/` — `results.parquet` and per-sequence `details/<model>/<key>.json`

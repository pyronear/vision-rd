# 🧪 Experiments

This directory contains R&D experiments for Pyronear wildfire smoke detection. Each experiment is a self-contained Python project with its own dependencies, data, and results.

## 🚀 Starting a New Experiment

```bash
cp -r template/ my-experiment-name/
cd my-experiment-name/
```

Then update `pyproject.toml` with the experiment name, description, and dependencies:

```bash
# Edit pyproject.toml (name, description, deps)
make install
```

## 📁 Experiment Template Structure

```
experiments/<experiment-name>/
├── README.md              # Objective, approach, results, how to reproduce
├── pyproject.toml         # Experiment deps (uv) + ruff config
├── uv.lock
├── .python-version        # Pinned Python version
├── .dvc/                  # Experiment-specific DVC config
├── .dvcignore
├── Makefile               # install, lint, format
├── src/                   # Source code
├── notebooks/             # Exploration notebooks
├── configs/               # Experiment configs (YAML)
├── scripts/               # Training/eval scripts
└── data/                  # Kedro-style layered data engineering
    ├── 01_raw/            # Immutable raw data as received
    ├── 02_intermediate/   # Cleaned, parsed, typed data
    ├── 03_primary/        # Domain-specific datasets ready for analysis
    ├── 04_feature/        # Engineered features for modeling
    ├── 05_model_input/    # Final train/val/test splits
    ├── 06_models/         # Trained model artifacts (weights, checkpoints)
    ├── 07_model_output/   # Predictions, scores, inference outputs
    └── 08_reporting/      # Plots, metrics, reports
```

## 🗂️ Kedro Data Layers

The `data/` directory follows [Kedro's data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention) with numbered layers that reflect the data processing pipeline:

| Layer | Purpose | Example |
|-------|---------|---------|
| `01_raw` | Immutable input data | Downloaded datasets, raw images |
| `02_intermediate` | Cleaned and parsed data | Filtered frames, parsed annotations |
| `03_primary` | Domain-specific datasets | Smoke/no-smoke labeled crops |
| `04_feature` | Engineered features | Extracted embeddings, optical flow |
| `05_model_input` | Train/val/test splits | Final tensors ready for training |
| `06_models` | Trained artifacts | Checkpoints, exported weights |
| `07_model_output` | Predictions and scores | Inference results, confidence maps |
| `08_reporting` | Communication artifacts | Plots, metrics tables, reports |

Data flows from lower to higher numbers. Raw data is never modified — all transformations produce new files in subsequent layers.

## 📏 Guidelines

See [GUIDELINES.md](GUIDELINES.md) for standards on tooling, reproducibility, and benchmarking.

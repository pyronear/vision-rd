# R&D Guidelines

Standards for all experiments under `experiments/`.

## Isolation

Each experiment is a self-contained unit with its own:
- `pyproject.toml` and `uv.lock` for dependencies
- `.python-version` for pinned Python version
- `.venv/` virtual environment (gitignored)
- `.dvc/` configuration for data tracking

Do not share dependencies between experiments. This ensures each is reproducible independently.

## Package Management

- Use **uv** exclusively for dependency management
- Pin Python version in `.python-version` (e.g., `3.11`)
- Pin dependency versions in `pyproject.toml` (e.g., `torch>=2.2,<2.3`)
- Use `uv run` to execute scripts within the project environment
- Run `uv sync` (or `make install`) after cloning or updating deps

## Code Quality

- Use **ruff** for both linting and formatting
- Configure ruff in `pyproject.toml` (the template provides a default config)
- Run `make lint` and `make format` before committing
- Recommended ruff rules: `["E", "F", "I", "W", "UP", "B", "SIM"]`

## Data Versioning

- Use **DVC** for all data and model artifacts
- Never commit large files (datasets, weights, images) directly to git
- Organize data using Kedro-style layers in `data/` (see [README.md](README.md))
- Track data with `dvc add data/` and commit the `.dvc` files to git
- Configure a DVC remote in `.dvc/config` using the convention: `s3://pyro-vision-rd/dvc/experiments/<experiment-name>/`

## Reproducibility

Every experiment must be reproducible from a clean checkout. Requirements:

- **Fixed random seeds**: Set and document seeds for all sources of randomness (Python, NumPy, PyTorch/TF)
- **Explicit configs**: Use YAML config files in `configs/` — no magic numbers in code
- **Pinned deps**: Always commit `uv.lock`
- **Logged hyperparameters**: Record all hyperparameters alongside metrics
- **Versioned data**: Use DVC hashes to track exact data versions used

## Experiment Tracking

Use DVC experiments or MLflow to track runs. For each experiment, record:

- Model architecture and key design choices
- Hyperparameters (learning rate, batch size, augmentations, etc.)
- Data version (DVC hash of the training data)
- Metrics (see Benchmarking below)
- Hardware used (GPU model, training time)

## Benchmarking

Use standardized metrics relevant to the Pyronear use case:

| Metric | Description |
|--------|-------------|
| Recall @ FPR | Detection recall at various false positive rates |
| Time-to-detection | Seconds from smoke onset to first alert |
| Inference latency | Milliseconds per frame (specify hardware) |
| Model size | Parameter count and FLOPs |

Compare against the current production baseline (YOLOv8 small) when applicable.

## Naming

- Experiment directories: **lowercase-kebab-case** (e.g., `temporal-smoke-classifier`, `background-subtraction-baseline`)
- Python packages/modules: **snake_case**
- Config files: descriptive names (e.g., `train_resnet50_lr1e3.yaml`)

## Documentation

Each experiment README must include:

1. **Objective** — What problem this project addresses
2. **Approach** — Method and architecture choices
3. **Data** — What data is used and how to obtain it
4. **Results** — Key metrics and comparison to baselines
5. **How to Reproduce** — Step-by-step instructions from clone to results

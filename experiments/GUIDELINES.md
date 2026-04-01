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

## Notebooks

Notebooks are for **exploration and visualization only**. Move any reusable logic into `src/` modules and import it from the notebook.

- **No committed outputs**: Always clear cell outputs before committing. The `nbstripout` git filter enforces this automatically — `make install` sets it up.
- **Linting**: `make lint` and `make format` cover both `.py` files and notebooks (via `nbqa ruff`).
- **Keep notebooks focused**: One analysis per notebook. Split large explorations into separate files.
- **Naming**: Use descriptive lowercase-kebab-case with an optional numeric prefix (e.g., `01-eda-smoke-crops.ipynb`, `02-model-comparison.ipynb`).

## Data Versioning

- Use **DVC** for all data and model artifacts
- Never commit large files (datasets, weights, images) directly to git
- Organize data using Kedro-style layers in `data/` (see [README.md](README.md))
- Track data with `dvc add data/` and commit the `.dvc` files to git
- Configure a DVC remote in `.dvc/config` using the convention: `s3://pyro-vision-rd/dvc/experiments/<experiment-name>/`

## Dataset Import

All experiments that use the Pyronear sequential dataset must import it from [`pyro-dataset`](https://github.com/pyronear/pyro-dataset) via `dvc import` with a pinned version tag. This ensures every experiment tracks exactly which dataset version it uses and where it came from.

### Standard import commands

```bash
# Train/val splits
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/train \
    -o data/01_raw/datasets/train --rev v2.2.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/val \
    -o data/01_raw/datasets/val --rev v2.2.0

# Test split (for leaderboard/evaluation)
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_test \
    -o data/01_raw/datasets/test --rev v2.2.0
```

### Conventions

- **Version tag**: Always use `--rev <tag>` (e.g. `v2.2.0`), never a branch name
- **Output path**: `data/01_raw/datasets/{train,val,test}`
- **Frozen imports**: DVC imports are frozen by default — they won't re-check the remote on `dvc repro`
- **Collaborators**: After cloning, just run `uv run dvc pull` to fetch data from the experiment's S3 remote
- **Commit `.dvc` files**: The resulting `.dvc` files (e.g. `data/01_raw/datasets/train.dvc`) must be committed to git

### Preprocessing (truncation, filtering, etc.)

If your experiment needs to preprocess the imported data (e.g. truncate sequences to N frames), import to `data/01_raw/datasets_full/` and add a pipeline stage that writes to `data/01_raw/datasets/`:

```bash
# Import full dataset
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/train \
    -o data/01_raw/datasets_full/train --rev v2.2.0
```

Then add a `truncate` stage in `dvc.yaml` (see template for example). This keeps downstream stages unchanged since they already depend on `data/01_raw/datasets/`.

## DVC Pipelines

Define your ML pipeline as stages in `dvc.yaml`. The template provides a commented-out scaffold — uncomment and adapt the stages you need.

### Structure

A typical pipeline has four stages: **prepare → split → train → evaluate**. Each stage declares its command, dependencies, parameters, and outputs so DVC can track lineage and skip unchanged steps.

### Conventions

- **Commands**: Always use `uv run python scripts/<stage>.py` to run within the project environment
- **CLI arguments**: Thread all inputs, outputs, and parameters directly to scripts via flags (e.g., `--input-dir`, `--seed ${train.seed}`). Scripts should not read paths or config files themselves — receive everything from the command line.
- **Parameters**: Store hyperparameters in `params.yaml` at the project root. Reference them in `dvc.yaml` with `${group.key}` interpolation and list them under the `params:` field so DVC tracks them.
- **Data paths**: Follow the Kedro-style layers in `data/` (raw → intermediate → model_input → models → reporting)
- **Metrics**: Declare metrics files with `cache: false` so they are always readable in the working tree (e.g., `data/08_reporting/metrics.json`)
- **Plots**: Use the `plots:` field for visualization outputs (e.g., `data/08_reporting/plots/`)

### Running the pipeline

```bash
uv run dvc repro          # run the full pipeline (skips up-to-date stages)
uv run dvc repro train    # run up to and including the train stage
uv run dvc params diff    # compare parameter changes
uv run dvc metrics show   # display current metrics
uv run dvc plots show     # render plots
```

### Experimenting with parameters

```bash
uv run dvc exp run -S train.learning_rate=0.001   # run with modified param
uv run dvc exp show                                # compare experiment results
uv run dvc exp apply <exp-name>                    # apply best experiment
```

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

# Per-Run Training Curves — Design

## Motivation

Each training run produces a PyTorch Lightning `CSVLogger` metrics file at
`data/06_models/<variant>/csv_logs/version_<N>/metrics.csv`. The file captures
per-step `train/loss` and per-epoch `val/{loss,accuracy,f1,precision,recall}`.
Today nothing renders these as human-readable plots — we only consume them via
TensorBoard (per-user, ephemeral) or DVC's built-in plots viewer (limited
styling). We want a static PNG per run that matches the visual quality of
Ultralytics' training summaries, committed to the reporting data layer so the
training dynamics of every variant are easy to inspect, compare visually, and
include in reports.

## Scope

In scope:
- A single PNG per training run summarizing loss and validation metrics over
  epochs, following an Ultralytics-style 2×3 grid.
- DVC wiring so the plot is produced as part of the pipeline, one stage per
  existing training variant.
- Unit tests covering the pure data-manipulation helpers.

Out of scope:
- Multi-variant comparison plots (already partially covered by
  `scripts/compare_variants.py`; different concern, will be addressed
  separately if needed).
- Per-step / smoothed curves (we aggregate train loss per epoch for cleaner
  shared epoch axis; noise reduction comes from aggregation).
- Live / in-training plotting. Plots are produced post-hoc from the CSV.

## Plot specification

- **Figure:** 2×3 grid, size 12×7 inches, 150 DPI, `constrained_layout=True`.
- **Style:** `plt.style.use("seaborn-v0_8-whitegrid")` (matplotlib-bundled, no
  `seaborn` dependency).
- **Subplot order (row-major):**
  - Row 1: `train/loss`, `val/loss`, `val/accuracy`
  - Row 2: `val/precision`, `val/recall`, `val/f1`
- **Per subplot:**
  - Line plot with small markers: `ax.plot(epoch, values, marker='.',
    linewidth=1.8)`.
  - Title: the metric name (e.g. `train/loss`).
  - X-axis label `epoch` on bottom row only.
  - Grid: `alpha=0.3`.
  - Best-epoch marker: vertical dashed gray line (`alpha=0.6`) at
    `argmax(val/f1)`, drawn on all 6 subplots so the best epoch lines up
    vertically across the figure.
- **Suptitle:** the variant name (passed via `--title`).

## Data flow

Input CSV columns: `epoch, step, train/loss, val/accuracy, val/f1, val/loss,
val/precision, val/recall`. Train rows have `train/loss` filled and val
columns empty; val rows are the inverse, logged once per epoch at epoch end.

1. `scripts/plot_training.py` receives `--csv-log-dir`, `--output-path`,
   `--title`.
2. Script discovers `version_*/metrics.csv` under the csv-log dir and selects
   the one with the highest version number.
3. Loads it into a `pandas.DataFrame`.
4. `aggregate_train_loss_per_epoch(df)` returns a `pd.Series` indexed by
   epoch:
   `df.dropna(subset=["train/loss"]).groupby("epoch")["train/loss"].mean()`.
5. `extract_val_metrics_per_epoch(df)` returns a DataFrame with the val rows:
   `df.dropna(subset=["val/f1"])[["epoch", "val/loss", "val/accuracy",
   "val/f1", "val/precision", "val/recall"]]`.
6. `best_epoch = val_df.loc[val_df["val/f1"].idxmax(), "epoch"]`.
7. Builds the 2×3 figure, plots each series, draws `axvline(best_epoch, ...)`
   on all subplots, sets suptitle, saves to `--output-path` (creating parent
   directories).
8. Prints a single `Wrote <path>` line and exits 0.

## Module & script layout

New file `src/smokeynet_adapted/training_plots.py` exposes three functions:

- `aggregate_train_loss_per_epoch(df: pd.DataFrame) -> pd.Series` — pure, no
  matplotlib.
- `extract_val_metrics_per_epoch(df: pd.DataFrame) -> pd.DataFrame` — pure, no
  matplotlib.
- `plot_training_curves(csv_path: Path, output_path: Path, title: str) -> None`
  — orchestrates: load CSV, call helpers, build figure, save PNG.

New file `scripts/plot_training.py`:
- Thin CLI wrapper.
- Args: `--csv-log-dir` (Path), `--output-path` (Path), `--title` (str).
- Picks the highest-numbered `version_*/metrics.csv` under `--csv-log-dir`.
- Calls `plot_training_curves(...)`.

All imports live at the top of each file per project convention.

## DVC wiring

Seven new stages, one per existing training variant (`mean_pool`, `gru`,
`gru_seed43`, `gru_seed44`, `gru_convnext`, `gru_finetune`,
`gru_convnext_finetune`). Example for `gru`:

```yaml
plot_training_gru:
  cmd: >-
    uv run python scripts/plot_training.py
    --csv-log-dir data/06_models/gru/csv_logs
    --output-path data/08_reporting/training/gru/training_curves.png
    --title gru
  deps:
    - scripts/plot_training.py
    - src/smokeynet_adapted/training_plots.py
    - data/06_models/gru/csv_logs
  plots:
    - data/08_reporting/training/gru/training_curves.png
```

Rationale for one stage per variant (rather than a `foreach` over variant
names): it matches the existing per-variant stage pattern (`train_*`,
`evaluate_*`) and keeps the diff localized when variants are added or
removed.

No new entries in `params.yaml`.

## Output layout

New top-level reporting subdirectory:

```
data/08_reporting/
  train/<variant>/...       # existing (train split evaluation)
  val/<variant>/...         # existing (val split evaluation)
  training/<variant>/       # NEW
    training_curves.png
```

`training/` sits as a sibling of `train/` and `val/`. These plots are a
training-run artifact (mixing train and val series) and don't fit under
either split dir.

## Error handling & edge cases

- **Missing CSV:** script raises `FileNotFoundError` with the path it looked
  for under `--csv-log-dir`.
- **No val rows yet** (e.g. training aborted mid-epoch 0): val subplots
  render as empty axes with a "no data yet" annotation via `ax.text(...)`;
  the train-loss subplot still renders; the best-epoch `axvline` is skipped.
- **Single-epoch run:** `axvline` at epoch 0 renders without special handling.
- **Multiple `version_*` directories:** pick the highest-numbered one (most
  recent run). This behavior is documented in `--help`.

## Testing

New file `tests/test_training_plots.py` covering the pure helpers. No
matplotlib calls exercised in CI.

- `test_aggregate_train_loss_per_epoch` — hand-built DataFrame with a mix of
  train and val rows across multiple epochs; assert returned Series equals
  expected per-epoch means.
- `test_aggregate_handles_missing_train_rows` — DataFrame containing only
  val rows; assert the function returns an empty Series without error.
- `test_extract_val_metrics_per_epoch` — mixed DataFrame; assert returned
  DataFrame has one row per epoch with the expected values and columns.
- `test_extract_handles_partial_val_rows` — a val-side row missing one of
  the metric columns is excluded (use `val/f1` as the sentinel column).

A smoke test that invokes `plot_training_curves` against a tiny fixture CSV
and just asserts the output PNG is created is optional and can be added if
desired.

## Dependencies

- `matplotlib` — already declared in `pyproject.toml`; no change.
- `pandas` — NOT currently available in the environment. Add it to
  `pyproject.toml` (`pandas>=2.0`) and refresh `uv.lock` as part of this
  work.

No other new third-party packages.

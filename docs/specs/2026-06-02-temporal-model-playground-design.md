# Temporal Model Playground — Design

**Date:** 2026-06-02
**Status:** Approved, pending implementation

## Context

The temporal-models repo now has several models behind heavier machinery: the
`temporal-model-leaderboard` runs full test-set evaluation with metrics, and the
`temporal-model-explorer` is a Streamlit app over a precomputed sequence store.
Neither is a good fit for the common ask "I just want to point a model at a
folder of frames and see what it decides."

This adds a small, isolated `playground` experiment that does exactly that: a
CLI to run a temporal model on a directory or list of images, plus a plain
Python example showing the same in code. Scope today is **bbox-tube-temporal
only**, structured so adding models later is cheap.

## Architecture

A new self-contained uv experiment at `experiments/temporal-models/playground/`,
copied from `experiments/template/` per `experiments/GUIDELINES.md`. It depends
on the shared libs and reuses the existing model contract — no new model code.

```
experiments/temporal-models/playground/
├── pyproject.toml          # name=playground; [project.scripts] playground = "playground.cli:main"
├── README.md               # Objective / Approach / Data (DVC) / How to use
├── Makefile, .python-version, .envrc, .gitignore, .gitattributes, .dvcignore
├── .dvc/config             # remote s3://pyro-vision-rd/dvc/experiments/playground/
├── uv.lock
├── src/playground/
│   ├── __init__.py
│   ├── cli.py              # argparse `run` subcommand + main()
│   └── core.py             # resolve_frames, resolve_model_package, max_probability, format_summary
├── examples/
│   └── run_on_frames.py    # plain, commented script: from_package → predict_sequence → inspect output
├── tests/
│   ├── __init__.py
│   └── test_core.py        # TDD for the pure helpers
└── data/01_raw/
    ├── models/bbox-tube-vit-dinov2/model.zip   # real package copied in (DVC-tracked)
    └── sample_sequences/
        ├── smoke-<seq>/  000_*.jpg ...  # ~2 smoke sequences (flattened, reindexed)
        └── fp-<seq>/     000_*.jpg ...  # ~2 false-positive sequences (flattened, reindexed)
```

### Dependencies (`pyproject.toml`)

```toml
dependencies = ["pyrocore", "bbox-tube-temporal"]

[tool.uv.sources]
pyrocore = { path = "../../../lib/pyrocore" }
bbox-tube-temporal = { path = "../../../lib/bbox-tube-temporal" }

[project.scripts]
playground = "playground.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/playground"]
```

`../../../` depth: `playground/` is at `experiments/temporal-models/playground/`,
and the libs are at repo-root `lib/` — three levels up. Verify against the
sibling `temporal-model-explorer/pyproject.toml` which uses the same depth.

## How the model is used (existing contract — do not reimplement)

The model already lives in `lib/bbox-tube-temporal/src/bbox_tube_temporal/model.py`:

```python
from bbox_tube_temporal.model import BboxTubeTemporalModel

model = BboxTubeTemporalModel.from_package(package_path, device=device)  # device=None → auto cuda/mps/cpu
out = model.predict_sequence(sorted_frame_paths)   # list[Path] → TemporalModelOutput
```

- `predict_sequence` (from `pyrocore.TemporalModel`) calls the default
  `load_sequence` to build `Frame`s (frame_id = stem, timestamp parsed from the
  Pyronear filename convention), then `predict`. The playground needs no custom
  loading.
- The model runs YOLO itself at inference (the package ships the YOLO model), so
  sample sequences only need **raw images** — no precomputed bboxes.
- The model decides on **frame order**, not absolute timestamps (tubes are built
  across consecutive frames; TTD is a frame index). So a `Frame.timestamp` of
  `None` — which is what the default `load_sequence` yields for non-timestamped
  filenames — is fine. The only requirement is that filename-sort order equals
  temporal order (see Data/DVC for how sample dirs guarantee this).
- `out` is a `pyrocore.TemporalModelOutput`: `is_positive: bool`,
  `trigger_frame_index: int | None`, `details: dict`.
- Calibrated probability, when present, is at
  `details["tubes"]["kept"][*]["probability"]` (mirrors
  `temporal_model_explorer.outcomes.max_probability`; replicate the 3-line helper
  locally rather than depending on the explorer).

## CLI

```
uv run playground run [--model NAME | --model-package PATH] [--device DEV] [--json] INPUT...
```

- **INPUT** (one or more positional args):
  - a single directory → sorted glob of `*.jpg`, `*.jpeg`, `*.png` inside it
    (no recursion);
  - or multiple file paths → used in the given order.
- **Model** (exactly one required):
  - `--model NAME` → resolves to `data/01_raw/models/NAME/model.zip`;
  - `--model-package PATH` → arbitrary `.zip`.
- **`--device`**: forwarded to `from_package` (default `None` → auto cuda → mps → cpu).
- **Output**: default human-readable summary; `--json` dumps the full
  `TemporalModelOutput` (via `dataclasses.asdict`, `json.dumps(..., default=str)`)
  to stdout.

Summary format:

```
SMOKE ✓   trigger=frame 4  (..._2018-07-27T22-44-14.jpg)
probability=0.87   frames=20   runtime=412ms
```

For a negative: `NO SMOKE ✗   frames=20   runtime=...ms` (no trigger/probability
lines, or shown as `n/a`). `trigger_frame_index` maps into the resolved frame
list to recover the filename.

## Core helpers (`src/playground/core.py`, pure & unit-tested)

- `resolve_frames(inputs: list[str]) -> list[Path]` — if a single existing
  directory, sorted glob of image extensions; else validate each path exists and
  return in given order. Raises a clear error on empty/missing.
- `resolve_model_package(model: str | None, model_package: Path | None, models_dir: Path) -> Path`
  — exactly one of the two; build/validate the path.
- `max_probability(details: dict) -> float | None`.
- `format_summary(out, frame_paths, runtime_ms) -> str`.

`cli.py` wires these to `BboxTubeTemporalModel.from_package(...)` and
`predict_sequence`, timing the call with `time.perf_counter`.

## Example (`examples/run_on_frames.py`)

A standalone, heavily-commented script (not a notebook — Python file was
requested) showing the happy path: load a package, run on a sample sequence
directory (and on an explicit frame list), and read `out.is_positive` /
`out.trigger_frame_index` / `out.details`. Defaults to a bundled
`data/01_raw/sample_sequences/smoke-<seq>/` path so it's copy-paste runnable
after `dvc pull`.

## Data / DVC

Both the model package and the sample sequences are **real files copied in from
sibling experiments** (they already exist locally, DVC-pulled in their home
projects) — not DVC placeholders.

- **Model package**: copy a real bbox-tube-temporal package into
  `data/01_raw/models/bbox-tube-vit-dinov2/model.zip`. Source:
  `experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip`
  (156M; the larger `gru_convnext_finetune/model.zip`, 247M, is an alternative
  and can be added the same way under a second `<name>/` dir). So
  `--model bbox-tube-vit-dinov2` works out of the box.

- **Sample sequences**: source from the **explorer sequence store**
  `experiments/temporal-models/temporal-model-explorer/data/03_primary/sequences/`.
  Each leaf `seq_<id>/` has a `meta.json` (carrying `label` ∈
  `smoke|fp|unknown`) and `images/detection_<id>.jpg`. Pick ~2 `smoke` and ~2
  `fp` sequences (the store has 97 smoke / 301 fp). For each, **read `meta.json`
  to get the temporal frame order**, then copy the images — flattened, with a
  zero-padded index prefix in meta order — into
  `data/01_raw/sample_sequences/{smoke,fp}-<seq>/NNN_detection_<id>.jpg`.
  - The index prefix guarantees the playground's filename-sort equals temporal
    order, independent of the original `detection_<id>` numbering. Keeping the
    original name as a suffix preserves provenance.
  - `meta.json` itself is **not** copied; the flattened image dir is all the CLI
    needs.

- Both tracked with `dvc add` under the experiment, remote
  `s3://pyro-vision-rd/dvc/experiments/playground/`. Collaborators run
  `dvc pull`.

- **Note for implementer**: the `cp`/reindex assembly is a real action done
  during implementation (the source files are present locally). But do **not**
  run `dvc pull` (user syncs DVC manually) and do **not** run `dvc add`/`dvc
  push` — document those commands for the user to run.

## Testing

- TDD on `core.py` helpers (pure, no torch/weights): dir-vs-paths resolution and
  sorting, model-package resolution (both flags, error cases), `max_probability`
  over representative details dicts, `format_summary` for positive/negative.
- A smoke test that the CLI argument parser accepts the documented forms.
- Real model inference is **not** unit-tested (needs DVC weights); the README
  "How to use" section is the documented end-to-end check:
  `dvc pull` → `uv run playground run --model bbox-tube-vit-dinov2 data/01_raw/sample_sequences/smoke-<seq>/`
  → `SMOKE ✓`. (Runnable locally without `dvc pull` since the files are copied in
  during implementation.)
- CI: lint (`ruff check`, `ruff format --check`) + `pytest tests/ -v`, auto-
  discovered like other experiments.

## Decisions made during brainstorming

- Scope limited to **bbox-tube-temporal** for now (extensible later).
- Console-script entry `playground` (so `uv run playground run …` works) rather
  than `python scripts/run.py`.
- Example code under `examples/` as a `.py` file rather than `notebooks/`.
- Sample images **flattened** (no `images/` subdir); **2 smoke + 2 fp** mix.
- Sample sequences sourced from the **explorer store** (frames reindexed into
  `meta.json` order) and a **real `model.zip`** copied from bbox-tube-temporal —
  both committed to DVC by the user, but copied in for real during
  implementation so the demo runs without a `dvc pull`.

## Non-goals

- No metrics/evaluation (that's the leaderboard).
- No UI/store (that's the explorer).
- No support for the other three models yet.
- No DVC pipeline (`dvc.yaml`) — this is a runner/demo, not a training pipeline.

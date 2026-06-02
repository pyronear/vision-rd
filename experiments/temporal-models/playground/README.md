# playground

Quickly run a temporal smoke model on a directory or sequence of frames.

## Objective

Try the `bbox-tube-temporal` temporal model on arbitrary frames without the
leaderboard/explorer machinery — point it at a folder of images (or a list of
files) and read the keep/discard decision.

## Approach

A thin wrapper over `bbox_tube_temporal.model.BboxTubeTemporalModel`
(`from_package(...).predict_sequence(frame_paths)`), exposed two ways:

- a `playground run` CLI (console script), and
- an `examples/run_on_frames.py` script showing the same in code.

Frames are taken in **filename order** when a single directory is given, or in
the **given order** when explicit file paths are passed. The model runs YOLO
itself, so only raw images are needed.

## Data

Both tracked with DVC (remote `s3://pyro-vision-rd/dvc/experiments/playground/`):

- **Model packages** — `data/01_raw/models/<name>/model.zip`. Ships with
  `bbox-tube-vit-dinov2`.
- **Sample sequences** — `data/01_raw/sample_sequences/{smoke,fp}-<seq>/`, a
  couple of smoke and false-positive sequences (flattened to a single dir of
  images, indexed in temporal order) for an instant try.

After cloning, fetch them with `uv run dvc pull`.

To add your own model package, drop a `model.zip` at
`data/01_raw/models/<name>/model.zip` and reference it with `--model <name>`.
Re-assemble sample sequences from the explorer store with
`scripts/build_sample_sequences.py`.

## How to use

```bash
make install

# Run on a bundled sample sequence (a single directory of frames):
uv run playground run --model bbox-tube-vit-dinov2 data/01_raw/sample_sequences/smoke-seq_40844/
# SMOKE ✓   trigger=frame 3  (003_detection_15206.jpg)
# probability=1.00   frames=30   runtime=559ms

# A false-positive sequence:
uv run playground run --model bbox-tube-vit-dinov2 data/01_raw/sample_sequences/fp-seq_40312/
# NO SMOKE ✗   frames=19   runtime=537ms

# Pass an explicit list of frame files instead of a directory (used in the
# given order). Let the shell expand a glob, or list real paths yourself:
uv run playground run --model bbox-tube-vit-dinov2 data/01_raw/sample_sequences/smoke-seq_40844/*.jpg

# Full JSON output and a forced device, against an arbitrary package:
uv run playground run --model-package data/01_raw/models/bbox-tube-vit-dinov2/model.zip \
    data/01_raw/sample_sequences/smoke-seq_40844/ --json --device cpu
```

`--model NAME` resolves to `data/01_raw/models/NAME/model.zip`; use
`--model-package PATH` for an arbitrary `.zip`. `--device` defaults to auto
(cuda → mps → cpu).

When passing files explicitly, give **real paths** (a shell glob like
`.../smoke-seq_40844/*.jpg` is easiest) — each path must exist or the run errors.

## Tracking new local data in DVC

```bash
uv run dvc add data/01_raw/models data/01_raw/sample_sequences
git add data/01_raw/models.dvc data/01_raw/sample_sequences.dvc data/01_raw/.gitignore
uv run dvc push
```

# Sequential Label Audit — Design

- **Date**: 2026-04-23
- **Experiment path**: `experiments/data-quality/sequential/`
- **Package name**: `data_quality_sequential`
- **Status**: design approved, pending implementation plan

## 1. Purpose

Use any `pyrocore.TemporalModel` as an oracle to surface sequences in the
pyro-dataset sequential split whose **sequence-level label** is likely wrong.
A sequence lives in either `wildfire/` (ground truth positive) or `fp/`
(ground truth negative); disagreements between the model's binary prediction
and the parent-folder label become review candidates.

Output is two FiftyOne review sets per `(model, split)`:

- **FP set**: model predicted positive, ground truth is `fp/`.
- **FN set**: model predicted negative, ground truth is `wildfire/`.

A human opens the FiftyOne app, walks the ranked list, and decides whether
each flagged sequence is a real label error or a true model mistake.

### Explicitly out of scope

- Frame-level bbox annotation errors (future `experiments/data-quality/frame-level/`).
- Consensus / voting across multiple models.
- Automatic writeback of corrections to `pyro-dataset`.

## 2. Folder layout

New category `experiments/data-quality/` parallel to `experiments/temporal-models/`.
Siblings planned:

```
experiments/data-quality/
  sequential/        # this experiment
  frame-level/       # future
```

Inside `sequential/`, the standard experiment layout from
`experiments/template/` (self-contained uv project, Kedro data layers, DVC
pipeline, ruff config).

## 3. Component breakdown

### `src/data_quality_sequential/`

- **`registry.py`** — `MODEL_REGISTRY: dict[str, tuple[str, str]]` mapping a
  `model_type` key to `(module_path, class_name)`, plus
  `load_model(model_type, package_path) -> TemporalModel`. Copied/adapted
  from `temporal_model_leaderboard.registry`. Starts with a single entry:

  ```python
  MODEL_REGISTRY = {
      "bbox-tube-temporal": (
          "bbox_tube_temporal.model",
          "BboxTubeTemporalModel",
      ),
  }
  ```

  Contract: every registered class must expose
  `classmethod from_package(path) -> Self`. Already satisfied by
  `BboxTubeTemporalModel`.

- **`dataset.py`** — sequence discovery + ground-truth extraction. Given a
  split root (e.g. `data/01_raw/datasets/train/`), yields:

  ```python
  @dataclass
  class SequenceRef:
      name: str
      split: str                 # "train" | "val" | "test"
      ground_truth: bool         # True iff parent dir == "wildfire"
      frame_paths: list[Path]    # sorted by filename
  ```

  Ground truth is inferred from the `wildfire/` vs `fp/` parent directory.
  Frame discovery mirrors `pyro_detector_baseline.data` and
  `temporal_model_leaderboard.dataset` — likely copied near-verbatim.

  (The initially-planned `camera: str | None` field was dropped at
  implementation time — reviewers can read the camera off `sequence_name`
  directly, and no downstream consumer needs it structured.)

- **`review.py`** — disagreement extraction. Given ground truth refs + a
  model's `predictions.json` for one split:
  - FP: `predicted_positive ∧ ¬ground_truth`
  - FN: `¬predicted_positive ∧ ground_truth`

  Both sets are emitted unranked — stable alphabetical order by sequence
  name. No ranking is applied; reviewers walk the list as is. Ranking
  (by score or by `trigger_frame_index`) can be added later if the
  unranked lists turn out to be unwieldy.

  Returns `ReviewSet` dataclasses consumed by the CSV + FiftyOne scripts.

### `scripts/`

- **`predict.py`** — loads one model package, runs
  `TemporalModel.predict_sequence` on every sequence of one split, writes
  `predictions.json` to `data/07_model_output/<model_name>/<split>/`.
  CLI: `--model-name`, `--model-type`, `--model-package`, `--split-dir`,
  `--output-dir`.

- **`build_review_sets.py`** — given predictions + ground truth for one
  `(model, split)`, writes to `data/08_reporting/<model_name>/<split>/`:
  - `fp_sequences.csv` (ranked)
  - `fn_sequences.csv` (ranked)
  - `summary.json` (counts, confusion matrix)
  - `review_manifest.json` (machine-readable list consumed by FiftyOne stage)

- **`build_fiftyone.py`** — reads a review manifest, creates two persistent
  FiftyOne datasets named `<model_name>_<split>_fp` and
  `<model_name>_<split>_fn`. Each sample = one sequence (represented as an
  image grid or an image group — implementation detail, decided during
  plan). Fields attached: ground truth label, model prediction,
  suspicion score, trigger frame index, camera name, split.

  Since FiftyOne datasets persist in a local mongo store rather than as
  files, the DVC output is a sentinel JSON at
  `data/fiftyone/<model_name>/<split>/datasets.json` listing the dataset
  names that were (re)created.

## 4. Data layout

```
data/
  01_raw/
    datasets/                                           # dvc import from pyro-dataset
      train.dvc
      val.dvc
      test.dvc
      train/{wildfire,fp}/<seq>/images/
      val/...
      test/...
    models/                                             # dvc import from bbox-tube-temporal
      bbox-tube-temporal-vit-dinov2-finetune.zip.dvc
  07_model_output/
    <model_name>/
      <split>/
        predictions.json                                # per-sequence raw output
  08_reporting/
    <model_name>/
      <split>/
        fp_sequences.csv
        fn_sequences.csv
        summary.json
        review_manifest.json
  fiftyone/
    <model_name>/
      <split>/
        datasets.json                                   # sentinel only
```

### Dataset imports

Following the standard pattern from `experiments/GUIDELINES.md`:

```bash
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/train \
    -o data/01_raw/datasets/train --rev <tag>
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/val \
    -o data/01_raw/datasets/val --rev <tag>
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_test \
    -o data/01_raw/datasets/test --rev <tag>
```

### Model imports

Packaged `.zip` archives are `dvc import`ed from the
`bbox-tube-temporal` experiment at a pinned git ref of *this* repo:

```bash
uv run dvc import <vision-rd-repo-url> \
    experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip \
    -o data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip \
    --rev <tag-or-branch>
```

**Prerequisite**: the referenced commit must have been
`dvc push`ed in the `bbox-tube-temporal` experiment before `dvc pull` here
can fetch the archive. The pinned rev is stored in the generated
`.zip.dvc` file; collaborators just `dvc pull`.

## 5. Pipeline (`dvc.yaml`)

Three stages, each using `foreach` over the `(model, split)` matrix:

```yaml
stages:
  predict:
    matrix:
      model: ${models}           # keys of params.yaml `models:`
      split: ${splits}
    cmd: >-
      uv run python scripts/predict.py
      --model-name ${item.model}
      --model-type ${models[${item.model}].model_type}
      --model-package data/01_raw/models/${item.model}.zip
      --split-dir data/01_raw/datasets/${item.split}
      --output-dir data/07_model_output/${item.model}/${item.split}
    deps:
      - scripts/predict.py
      - src/data_quality_sequential
      - data/01_raw/datasets/${item.split}
      - data/01_raw/models/${item.model}.zip
    outs:
      - data/07_model_output/${item.model}/${item.split}

  build_review_sets:
    # ...analogous: inputs = predictions + datasets, output = 08_reporting...

  build_fiftyone:
    # ...analogous: input = review_manifest, output = fiftyone/.../datasets.json...
```

Exact DVC `foreach`/matrix syntax will be finalized during plan writing —
the leaderboard's `dvc.yaml` is the reference.

## 6. `params.yaml`

Initially planned as a registration point for models/splits. At
implementation time this was dropped: the neighboring
`temporal-model-leaderboard` uses per-model stage blocks with no
`params.yaml`, and copying that convention keeps adding a model as one
block of `dvc.yaml` edits + one registry entry (no hidden wiring
between `params.yaml` and `dvc.yaml`).

Adding another model variant today: `dvc import` its `.zip`, add one
`MODEL_REGISTRY` entry if needed, and copy-paste the three stage blocks
in `dvc.yaml` with the new model-name substituted. See the experiment's
`README.md` § "Adding another model".

## 7. Testing

- `tests/test_registry.py` — `load_model("bbox-tube-temporal", fixture_zip)`
  returns a working `TemporalModel`. Uses a small fixture zip (or mock) to
  keep the test fast.
- `tests/test_dataset.py` — on a synthetic split directory with a handful
  of `wildfire/` and `fp/` sequences, `SequenceRef` iteration returns the
  correct ground-truth booleans and sorted frame paths.
- `tests/test_review.py` — hand-crafted ground-truth + prediction pairs:
  FP/FN selection is correct and output order is deterministic
  (alphabetical by sequence name).

End-to-end validation is via `dvc repro` on real data — not a unit test.

## 8. Conventions recap

- **Ranking**: none. Both FP and FN sets are emitted unranked in stable
  alphabetical order by sequence name. Can be revisited if the lists
  prove unwieldy.
- **Naming**: `model_name` follows the packaged-variant naming already
  used by `temporal-model-leaderboard` (e.g.
  `bbox-tube-temporal-vit-dinov2-finetune`). `model_type` is the registry
  key matching the experiment folder name (e.g. `bbox-tube-temporal`).
- **Paths**: every artifact is keyed by `<model_name>/<split>`. Adding or
  removing a model never touches another model's data.

## 9. Open questions deferred to the plan

- Exact FiftyOne sample representation for a sequence: image group vs
  image grid vs per-frame samples with a shared `sequence_id` field.
  Leaderboard's `build_fiftyone_errors.py` has a pattern to copy.
- Whether to export the FiftyOne dataset to a portable format (e.g.
  `fiftyone.core.dataset.Dataset.export`) in addition to the mongo-backed
  form, for sharing review sets across machines.
- If the unranked review lists turn out to be too long to walk
  comfortably, introduce ranking (either by `trigger_frame_index` or by
  a cross-model `details["score"]` convention).

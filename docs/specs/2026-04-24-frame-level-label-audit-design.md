# Frame-Level Label Audit — Design

- **Date**: 2026-04-24
- **Experiment path**: `experiments/data-quality/frame-level/`
- **Package name**: `data_quality_frame_level`
- **Status**: design approved, pending implementation plan

## 1. Purpose

Use the production YOLO smoke detector (`pyronear/yolo11s_nimble-narwhal_v6.0.0`)
as a single-frame oracle to surface likely **frame-level label errors** in
pyro-dataset's flat YOLO split (`yolo_train_val` + `yolo_test`).

For every image, compare YOLO detections against the per-image `.txt` ground
truth and surface three categories for human review:

- **FP** — model predicts a bbox where no GT bbox overlaps it
  (likely a missing annotation, or a real false alarm).
- **FN** — GT bbox has no overlapping model prediction
  (likely a spurious annotation, or a genuine model miss).
- **Bbox disagreement** — model and GT both have bboxes in the same region but
  IoU is below the match threshold (sloppy annotation, or poor localization).

Output is one FiftyOne dataset per `(model, split)` with GT and prediction
bboxes overlaid and FiftyOne's standard `evaluate_detections` schema
attached, so reviewers use the built-in Evaluation panel to filter by
error type.

### Explicitly out of scope

- Sequence-level (folder) labels — covered by `experiments/data-quality/sequential/`.
- Temporal smoothing / alarm aggregation — this is single-frame detection.
- Consensus / voting across multiple YOLO variants.
- Automatic writeback of corrections to `pyro-dataset`.

## 2. Folder layout

New sibling under the existing data-quality category:

```
experiments/data-quality/
  sequential/        # existing
  frame-level/       # this experiment
```

Inside `frame-level/`, the standard experiment layout from
`experiments/template/` — self-contained uv project, Kedro data layers,
DVC pipeline, ruff config.

## 3. Component breakdown

### `src/data_quality_frame_level/`

- **`dataset.py`** — split discovery + GT parsing. Given a split root
  (`data/01_raw/datasets/<split>/`), yields:

  ```python
  @dataclass
  class FrameRef:
      stem: str                    # filename without extension
      image_path: Path
      label_path: Path             # may exist with empty content
      gt_bboxes: list[BBox]        # empty iff label file is empty or absent
  ```

  YOLO label lines are parsed as `class cx cy w h` (normalized floats in
  `[0, 1]`). Empty `.txt` → empty `gt_bboxes` (this is how `fp/`-origin
  frames land in the flat split).

- **`inference.py`** — thin wrapper around
  `ultralytics.YOLO(model_path).predict(..., conf, device=0)` returning
  per-image prediction lists: `[(bbox_xyxy_px, conf, class), ...]`.
  Batched inference over a split.

- **`fiftyone_build.py`** — build one FiftyOne dataset per `(model, split)`:
  - Each sample = one image.
  - Field `ground_truth: fo.Detections` — normalized top-left xywh, converted
    from YOLO center-xywh.
  - Field `predictions: fo.Detections` — each prediction carries its `conf`.
  - Call `dataset.evaluate_detections(pred_field="predictions",
    gt_field="ground_truth", eval_key="eval", iou=<iou_thresh>)`.
    This populates per-detection `eval` (`tp` / `fp` / `fn`) and per-sample
    `eval_tp`, `eval_fp`, `eval_fn` counts — everything FiftyOne's
    Evaluation panel needs.
  - Write per-split aggregate counts (TP/FP/FN, precision/recall, mAP if
    available) to `08_reporting/<model>/<split>/summary.json`.

No `pyrocore.TemporalModel` dependency — this is pure single-frame object
detection, so the temporal abstraction doesn't fit.

### `scripts/`

- **`prepare.py`** — `hf_hub_download(repo_id, filename)` → writes
  `data/01_raw/models/<model_name>.pt`. Skips the download if the file
  already exists. CLI: `--hf-repo`, `--hf-filename`, `--output`.

- **`predict.py`** — loads a `.pt` model, runs YOLO on every image in one
  split, writes `predictions.json` to
  `data/07_model_output/<model_name>/<split>/`. CLI: `--model-name`,
  `--model-path`, `--split-dir`, `--conf-thresh`, `--output-dir`.

  `predictions.json` shape: `{image_rel_path: [{bbox_xyxy_px, conf, class}, ...]}`.

- **`build_fiftyone.py`** — reads GT (from split dir) + `predictions.json`,
  creates a persistent FiftyOne dataset, runs `evaluate_detections`, and
  writes a sentinel at `data/fiftyone/<model_name>/<split>/dataset.json`
  plus a `summary.json` to `08_reporting/<model_name>/<split>/`.

- **`launch_fiftyone.py`** — dev convenience (copied from `sequential/`),
  opens the FiftyOne app in the foreground. Invoked by `make fiftyone`.

## 4. Data layout

```
data/
  01_raw/
    datasets/                                  # dvc import from pyro-dataset
      train.dvc
      val.dvc
      test.dvc
      {train,val,test}/images/*.jpg
      {train,val,test}/labels/*.txt
    models/
      <model_name>.pt                          # downloaded by prepare, not dvc-tracked
  07_model_output/
    <model_name>/
      {train,val,test}/
        predictions.json
  08_reporting/
    <model_name>/
      {train,val,test}/
        summary.json                           # counts, precision/recall, mAP
  fiftyone/
    <model_name>/
      {train,val,test}/
        dataset.json                           # sentinel; actual dataset in mongo
```

### Dataset imports

From pyro-dataset @ `v3.0.0`:

```bash
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/yolo_train_val/images/train \
    -o data/01_raw/datasets/train/images --rev v3.0.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/yolo_train_val/labels/train \
    -o data/01_raw/datasets/train/labels --rev v3.0.0
# ...and analogous for val (from yolo_train_val) and test (from yolo_test).
```

Exact target paths inside pyro-dataset may need adjustment at
implementation time — verify against the repo tag before committing the
`.dvc` files. The merged `yolo_train_val` has `images/{train,val}` and
`labels/{train,val}` subtrees; `yolo_test` has `images/` and `labels/` at
its root.

### Model download

`prepare` stage pulls the `.pt` directly from Hugging Face via
`hf_hub_download`. No `dvc import` — the HF repo ID pins the version, and
the `.pt` is regenerated from HF on each `dvc repro prepare` rather than
stored in DVC cache. The exact `hf_filename` for the `.pt` asset is
resolved at implementation time from the HF repo contents.

## 5. Pipeline (`dvc.yaml`)

Three stages. `prepare` is keyed by `model` only; `predict` and
`build_fiftyone` are keyed by `(model, split)`:

```yaml
stages:
  prepare:
    matrix:
      model: ${models}
    cmd: >-
      uv run python scripts/prepare.py
      --hf-repo ${models[${item.model}].hf_repo}
      --hf-filename ${models[${item.model}].hf_filename}
      --output data/01_raw/models/${item.model}.pt
    outs:
      - data/01_raw/models/${item.model}.pt:
          cache: false

  predict:
    matrix:
      model: ${models}
      split: [train, val, test]
    cmd: >-
      uv run python scripts/predict.py
      --model-name ${item.model}
      --model-path data/01_raw/models/${item.model}.pt
      --split-dir data/01_raw/datasets/${item.split}
      --conf-thresh ${models[${item.model}].conf_thresh}
      --output-dir data/07_model_output/${item.model}/${item.split}
    deps:
      - scripts/predict.py
      - src/data_quality_frame_level
      - data/01_raw/datasets/${item.split}
      - data/01_raw/models/${item.model}.pt
    outs:
      - data/07_model_output/${item.model}/${item.split}

  build_fiftyone:
    matrix:
      model: ${models}
      split: [train, val, test]
    cmd: >-
      uv run python scripts/build_fiftyone.py
      --dataset-name dq-frame_${item.model}_${item.split}
      --split-dir data/01_raw/datasets/${item.split}
      --predictions data/07_model_output/${item.model}/${item.split}/predictions.json
      --iou-thresh ${models[${item.model}].iou_thresh}
      --sentinel data/fiftyone/${item.model}/${item.split}/dataset.json
      --summary data/08_reporting/${item.model}/${item.split}/summary.json
    deps:
      - scripts/build_fiftyone.py
      - src/data_quality_frame_level
      - data/01_raw/datasets/${item.split}
      - data/07_model_output/${item.model}/${item.split}
    outs:
      - data/fiftyone/${item.model}/${item.split}/dataset.json
      - data/08_reporting/${item.model}/${item.split}/summary.json
```

Matrix syntax to be finalized against the sequential and leaderboard
experiments during plan writing.

## 6. `params.yaml`

Model-specific parameters live here, keyed by `model_name`. Adding a new
YOLO variant = one params block + one dvc matrix entry, nothing else:

```yaml
models:
  yolo11s-nimble-narwhal:
    hf_repo: pyronear/yolo11s_nimble-narwhal_v6.0.0
    hf_filename: <pt filename in HF repo, resolved at implementation time>
    conf_thresh: 0.35       # per-detection YOLO confidence
    iou_thresh: 0.5         # evaluate_detections match IoU
```

### Threshold rationale

- **`conf_thresh: 0.35`** — matches the aggregated alarm threshold used in
  production (`pyro-engine`). In production this applies to the
  temporally-smoothed score, not to each raw YOLO detection; here it is
  applied per-detection because we have no temporal layer. Effect: only
  high-confidence detections produce flags, keeping the review backlog
  tractable. Lower (e.g. production YOLO per-detection `0.05`) would flood
  reviewers with low-confidence false alarms.
- **`iou_thresh: 0.5`** — standard object-detection match threshold.
  Detections matched at IoU ≥ 0.5 count as TP; below-threshold pairs show
  up as FP + FN (which is what we want to surface as "disagreement").

## 7. FiftyOne representation

- **One dataset per `(model, split)`** named `dq-frame_<model>_<split>`.
- Each sample = one image; fields:
  - `ground_truth: fo.Detections` — parsed from `.txt`.
  - `predictions: fo.Detections` — from YOLO, `confidence` per detection.
  - `eval_tp`, `eval_fp`, `eval_fn` — populated by `evaluate_detections`.
  - Per-detection `eval` label — `tp` / `fp` / `fn`.
- Reviewers open the FiftyOne Evaluation panel, filter by error type,
  and see both overlays side by side.
- Bbox conversion is the one tricky step: YOLO normalized
  `(cx, cy, w, h)` → FiftyOne normalized top-left
  `(x=cx−w/2, y=cy−h/2, w, h)`.

## 8. Testing

- `test_dataset.py` — synthetic split directory (a handful of `.jpg` with
  matching `.txt`, some empty, some multi-line). Assertions:
  - `FrameRef` iteration returns every image.
  - Empty `.txt` → empty `gt_bboxes`.
  - Multi-line `.txt` → correct count and parsed values.
- `test_fiftyone_build.py` — pure conversion test on
  YOLO-center-xywh → FiftyOne-top-left-xywh with a handful of edge cases
  (centered bbox, corner bbox). No live FiftyOne mongo.

End-to-end validation is `dvc repro` on real data — not a unit test.

## 9. Conventions recap

- **Model**: narwhal only to start
  (`pyronear/yolo11s_nimble-narwhal_v6.0.0`). Pattern scales to additional
  YOLO variants by adding one `params.yaml` entry.
- **Naming**: `model_name` is the `params.yaml` key
  (e.g. `yolo11s-nimble-narwhal`). FiftyOne datasets are
  `dq-frame_<model_name>_<split>` — `dq-frame` prefix mirrors the
  sequential experiment's `dq-seq`.
- **Paths**: every artifact is keyed by `<model_name>/<split>`. Adding or
  removing a model never touches another model's data.
- **Data layers**: Kedro-style under `data/` (`01_raw` → `07_model_output`
  → `08_reporting` → `fiftyone/`), matching `sequential/`.

## 10. Known limitations

- **Oracle was trained on this data.** Narwhal was trained on
  `yolo_train_val` (likely at an earlier pyro-dataset tag than the
  `v3.0.0` we audit). On the training split, it will "agree" with any
  label it memorized — including incorrect ones — so FP/FN/disagreement
  flags understate the true label-error rate on train. The val/test
  findings are more trustworthy in that regard, and even on train the
  flags are still useful signal (YOLO doesn't memorize perfectly).
  Keep this in mind when reading per-split counts.

## 11. Open questions deferred to the plan

- Exact `hf_filename` for the narwhal `.pt` asset — to be read off the
  Hugging Face repo during implementation.
- Exact path layout inside pyro-dataset's `yolo_train_val` / `yolo_test`
  at tag `v3.0.0` — verify before committing the `.dvc` imports.
- Whether `summary.json` should include mAP
  (`evaluate_detections` supports it) or only the simpler TP/FP/FN counts
  and precision/recall.
- If `dvc import` can't directly target sub-paths like
  `yolo_train_val/images/train`, we may need to import the whole
  `yolo_train_val` directory and compute the split views in a small
  prepare-style stage. Confirm against the existing sequential experiment
  imports.

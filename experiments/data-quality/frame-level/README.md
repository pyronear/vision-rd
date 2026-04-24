# data-quality/frame-level

Use a single-frame YOLO oracle (`pyronear/yolo11s_nimble-narwhal_v6.0.0`)
to surface **frame-level** label errors in pyro-dataset's flat YOLO
split (`yolo_train_val` + `yolo_test`):

- **FP** — YOLO predicts a bbox where the `.txt` ground truth has no
  overlapping box (likely a missing annotation).
- **FN** — GT bbox has no overlapping YOLO prediction (likely a
  spurious annotation, or a genuine model miss).

Every YOLO detection above the production per-detection threshold
(`0.05`) is retained, so reviewers can dynamically tune the confidence
filter live in the FiftyOne sidebar without rebuilding anything.

## Design

See [`docs/specs/2026-04-24-frame-level-label-audit-design.md`](docs/specs/2026-04-24-frame-level-label-audit-design.md).

## Pipeline

```
prepare  →  predict  →  build_fiftyone
 (per model)    (per model × split)      (per model × split)
```

Single DVC pipeline parameterized by `models:` in `params.yaml` — each
stage is `foreach`-expanded over the dict, producing per-model stages
named `<stage>@<model-name>`. Adding a new YOLO variant is one block in
`params.yaml` and nothing else.

## How to reproduce

```bash
cd experiments/data-quality/frame-level
make install

# Fetch the dvc-imported datasets via your usual `dvc pull` workflow.

# Full pipeline (train + val + test):
uv run dvc repro
```

Outputs:

- `data/07_model_output/<model>/<split>/predictions.json` — every YOLO detection ≥ `conf_thresh`.
- `data/08_reporting/<model>/<split>/summary.json` — TP/FP/FN counts, precision/recall, and review-queue sample sizes.
- Persistent FiftyOne datasets in the local mongo store, named `dq-frame_<model>_<split>`, each with two saved views (`fp-by-confidence`, `fn-by-area`).

## How to review the errors

The review workflow is built around **saved views** persisted on each FiftyOne dataset. Each dataset (`dq-frame_<model>_<split>`) carries two saved views with identical names across splits, so switching splits in the UI preserves the review workflow.

### Launch the app

Three `make` targets, all open `http://localhost:5151`:

| Command | What it opens |
|---|---|
| `make fiftyone-fp` | `dq-frame_yolo11s-nimble-narwhal_val` with the `fp-by-confidence` saved view applied. FP predictions above `review_conf_thresh` (0.35), sorted by predicted confidence descending — highest-confidence false positives first. |
| `make fiftyone-fn` | Same dataset with the `fn-by-area` saved view applied. Samples with any missed GT bbox, sorted by GT bbox area descending — largest missed annotations first. |
| `make fiftyone` | Alias for `fiftyone-fp`. |

The launch script forces label + confidence overlays on every bbox (both grid and expanded views).

### Switch between splits without losing the view

Both saved views use the same name on train, val, and test.

1. In the FiftyOne left sidebar, click the **dataset selector** (top of sidebar) and pick `dq-frame_yolo11s-nimble-narwhal_train` (or `_test`).
2. Below the dataset selector, open the **Saved views** dropdown and pick `fp-by-confidence` (or `fn-by-area`) — the filter + sort apply instantly.

This keeps the review experience consistent across splits: one click to change split, one click to re-apply the queue.

### Browse a specific split directly from the CLI

```bash
uv run --group explore python scripts/launch_fiftyone.py \
    --dataset dq-frame_yolo11s-nimble-narwhal_test \
    --kind fp
```

`--kind` accepts `fp`, `fn`, or `all`.

### Interpret the overlays

Every sample shows both fields side-by-side:

- `ground_truth` (blue) — human-labeled bboxes from the `.txt` file.
- `predictions` (teal) — YOLO detections with `confidence` displayed on the overlay.

Each detection has an `eval` attribute (`tp` / `fp` / `fn`), readable in the right-hand sample panel (click a sample to expand). Matched pairs also carry an `eval_iou` value.

We intentionally do **not** hide TP bboxes when filtering for FP/FN — both are shown so the review context is accurate. If you only see one bbox per sample in the grid thumbnail, expand it (click the sample) to see the full eval detail in the sidebar.

### Tune the review filters live (no rebuild needed)

All `predictions ≥ 0.05` are stored in the FiftyOne dataset. The saved view's `confidence ≥ 0.35` filter is just a default — you can override it without re-running the pipeline:

1. In the left sidebar, expand the `predictions` field group.
2. Drag the **confidence range slider** to include lower- or higher-confidence detections.
3. Combine with the `eval` filter (keep `fp` checked, untick `tp`/`fn`) if you want to filter within the currently-visible samples.

Changes are session-local and don't modify the saved views. If you want the new default to stick, update `review_conf_thresh` in `params.yaml` and rerun — only the three `build_fiftyone_*` stages will re-run (predictions are cached):

```bash
uv run dvc repro
```

### Workflow summary

```
dvc repro             # generate datasets + saved views (one-time, or after param changes)
  ↓
make fiftyone-fp      # open FP queue, walk highest-confidence first
  ↓
sidebar: switch split → pick 'fp-by-confidence' saved view
  ↓
sidebar: slide confidence range / filter by eval to tune
  ↓
make fiftyone-fn      # same flow for missed annotations
```

## Results

Config: `conf_thresh=0.05` (per-detection YOLO threshold), `iou_thresh=0.05`, `review_conf_thresh=0.35` (FP queue default).

| Split | Samples | Raw TP | Raw FP | Raw FN | FP review samples | FN review samples |
|-------|---------|--------|--------|--------|-------------------|-------------------|
| train | 14,767  | 14,224 | 14,375 |   436  |               939 |               430 |
| val   |  1,426  |  1,286 |  1,557 |   116  |               115 |               113 |
| test  |  2,640  |  1,476 |  1,846 |    88  |               220 |                79 |

**Raw counts** are bbox-level and include every candidate detection above `conf_thresh=0.05`. **Review samples** are image-level — the number of images in each saved view that a reviewer actually walks through (FP queue filters to `confidence ≥ review_conf_thresh`).

The IoU threshold is deliberately lenient (0.05 vs. the COCO-standard 0.5) — this is a label-quality audit, not a detection-evaluation benchmark. Partial-but-clearly-same-object overlaps count as TP so reviewers are only shown frames where the model and GT genuinely disagree about the *existence* of smoke.

Train review queues are larger and noisier — see Caveats.

## Data imports

`train`, `val`, `test` are imported from
[`pyro-dataset`](https://github.com/pyronear/pyro-dataset) at tag
`v3.0.0`:

```bash
# train + val come from yolo_train_val (flat under images/ and labels/)
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/yolo_train_val/images/train \
    -o data/01_raw/datasets/train/images --rev v3.0.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/yolo_train_val/labels/train \
    -o data/01_raw/datasets/train/labels --rev v3.0.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/yolo_train_val/images/val \
    -o data/01_raw/datasets/val/images --rev v3.0.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/yolo_train_val/labels/val \
    -o data/01_raw/datasets/val/labels --rev v3.0.0

# test comes from yolo_test, which has an extra test/ level under images/ and labels/
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/yolo_test/images/test \
    -o data/01_raw/datasets/test/images --rev v3.0.0
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/yolo_test/labels/test \
    -o data/01_raw/datasets/test/labels --rev v3.0.0
```

Model weights are downloaded fresh from Hugging Face by the `prepare`
stage — not tracked by DVC.

## Adding another YOLO variant

1. Add an entry keyed by the model name to `models:` in `params.yaml`:

   ```yaml
   models:
     yolo11s-nimble-narwhal:
       hf_repo: pyronear/yolo11s_nimble-narwhal_v6.0.0
       hf_filename: best.pt
       conf_thresh: 0.05
       review_conf_thresh: 0.35
       iou_thresh: 0.05
     <new-model-name>:
       hf_repo: <hf-org/new-model-repo>
       hf_filename: <pt-filename-in-repo>
       conf_thresh: 0.05
       review_conf_thresh: 0.35
       iou_thresh: 0.05
   ```

2. `uv run dvc repro` — only the new model's stages run (each DVC stage
   is `foreach`-expanded over `models:`, producing per-model stages
   named `<stage>@<model-name>`).

No changes to `dvc.yaml` needed. Because stages are keyed by model
name (not positional index), reordering entries in `params.yaml` does
not invalidate existing stages.

### Threshold parameters per model

| Param | Meaning | Effect of changing it |
|---|---|---|
| `conf_thresh` | YOLO per-detection confidence floor at inference time. | Re-runs `predict_*` (GPU, slow) and downstream `build_fiftyone_*`. Lower → more candidates retained. |
| `iou_thresh` | IoU threshold for TP/FP/FN assignment via `evaluate_detections`. | Re-runs only `build_fiftyone_*` (fast). Lower → partial-overlap near-matches count as TP. |
| `review_conf_thresh` | Confidence floor applied to the **FP saved view**. | Re-runs only `build_fiftyone_*`. Override live in the FiftyOne sidebar without rebuilding. |

## Caveats

- **Oracle was trained on this data.** Narwhal was trained on
  `yolo_train_val` (likely at an earlier pyro-dataset tag than the
  `v3.0.0` we audit). On the training split the model will "agree" with
  any label it memorized — including incorrect ones — so flags on train
  understate the true label-error rate. Val/test findings are more
  trustworthy. See §10 of the design spec for details.
- **Narwhal v6.0.0 runs at `conf=0.05` per-detection and a temporal
  smoothing threshold of `0.35` in production.** Here there's no
  temporal layer, so we use `conf=0.05` at inference (retain everything)
  and apply the `0.35` floor to the FP review queue instead, which
  approximates the production alarm gate at the single-frame level.

## Layout

```
data/
  01_raw/
    datasets/{train,val,test}/{images,labels}.dvc   # dvc-imported
    models/
      <model-name>.pt                               # downloaded; not dvc-tracked
  07_model_output/<model-name>/<split>/
    predictions.json
  08_reporting/<model-name>/<split>/
    summary.json
  fiftyone/<model-name>/<split>/
    dataset.json                                    # sentinel; actual dataset in mongo
```

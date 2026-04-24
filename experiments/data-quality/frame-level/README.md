# data-quality/frame-level

Use a single-frame YOLO oracle (`pyronear/yolo11s_nimble-narwhal_v6.0.0`)
to surface **frame-level** label errors in pyro-dataset's flat YOLO
split (`yolo_train_val` + `yolo_test`):

- **FP** — YOLO predicts a bbox where the `.txt` ground truth has no
  overlapping box (likely a missing annotation).
- **FN** — GT bbox has no overlapping YOLO prediction (likely a
  spurious annotation, or a genuine model miss).
- **Bbox disagreement** — matched GT + pred with IoU below the match
  threshold (sloppy annotation or poor localization).

Output is one FiftyOne dataset per split with GT + predictions attached.
Reviewers use the built-in Evaluation panel to filter by FP/FN and walk
through flagged frames.

## Design

See [`docs/specs/2026-04-24-frame-level-label-audit-design.md`](docs/specs/2026-04-24-frame-level-label-audit-design.md).

## Pipeline

```
prepare  →  predict  →  build_fiftyone
 (per model)    (per model × split)      (per model × split)
```

Single `matrix`-ed DVC pipeline parameterized by `models` in
`params.yaml`. Adding a new YOLO variant is one block in `params.yaml`
and nothing else.

## How to reproduce

```bash
cd experiments/data-quality/frame-level
make install

# Fetch the dvc-imported datasets via your usual `dvc pull` workflow.

# Full pipeline (train + val + test):
uv run dvc repro
```

Reports land in `data/08_reporting/<model>/<split>/summary.json`;
FiftyOne datasets named `dq-frame_<model>_<split>` are created in the
local FiftyOne mongo store.

Browse the review sets:

```bash
make fiftyone
```

Opens the FiftyOne app (default: http://localhost:5151). Use the
dataset selector in the sidebar to switch splits, and the Evaluation
panel to filter by FP/FN.

## Results (narwhal @ conf=0.35, IoU=0.5)

| Split | Samples | TP | FP | FN | Precision | Recall |
|-------|---------|----|----|----|-----------|--------|
| train | 14,767  | 11,825 | 1,842 | 2,835 | 0.865 | 0.807 |
| val   | 1,426   | 883    | 277   | 519   | 0.761 | 0.630 |
| test  | 2,640   | 1,155  | 382   | 409   | 0.751 | 0.738 |

Train numbers understate the true label-error rate — see Caveats.

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
       conf_thresh: 0.35
       iou_thresh: 0.5
     <new-model-name>:
       hf_repo: <hf-org/new-model-repo>
       hf_filename: <pt-filename-in-repo>
       conf_thresh: 0.35
       iou_thresh: 0.5
   ```

2. `uv run dvc repro` — only the new model's stages run (each DVC stage
   is `foreach`-expanded over `models:`, producing per-model stages
   named `<stage>@<model-name>`).

No changes to `dvc.yaml` needed. Because stages are keyed by model
name (not positional index), reordering entries in `params.yaml` does
not invalidate existing stages.

## Caveats

- **Oracle was trained on this data.** Narwhal was trained on
  `yolo_train_val` (likely at an earlier pyro-dataset tag than the
  `v3.0.0` we audit). On the training split the model will "agree" with
  any label it memorized — including incorrect ones — so FP/FN counts
  on train understate the true label-error rate. Val/test findings are
  more trustworthy. See §10 of the design spec for details.
- **`conf=0.35` is the production aggregated-alarm threshold**, not the
  production YOLO per-detection threshold (which is 0.05). Here it is
  applied per raw detection because there is no temporal smoothing —
  chosen to keep review noise low. If you want to see low-confidence
  detections, drop `conf_thresh` in `params.yaml`.

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

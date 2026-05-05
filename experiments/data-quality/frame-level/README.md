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
filter live in the review app without rebuilding anything.

## Quickstart — review session

```bash
cd experiments/data-quality/frame-level
make install                  # once per checkout

# Generate predictions from the YOLO oracle:
uv run dvc repro

# Start a review session:
make review-app               # opens http://localhost:8000

# End of session — emit the patch + push:
make review-export
uv run dvc add data/09_review data/10_export && uv run dvc push
git add data/09_review.dvc data/10_export.dvc data/.gitignore
git commit -m "review: bbox corrections + export"
```

## Design

See [`docs/specs/2026-04-24-frame-level-label-audit-design.md`](docs/specs/2026-04-24-frame-level-label-audit-design.md).

## Pipeline

```
prepare  →  predict
 (per model)    (per model × split)
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

## Bbox-editing review app

A browser-based review workflow that **edits GT bboxes inline** instead
of just tagging frames. Specs:
[`docs/specs/2026-05-05-review-app-design.md`](docs/specs/2026-05-05-review-app-design.md)
and [`docs/specs/2026-05-05-export-flow-design.md`](docs/specs/2026-05-05-export-flow-design.md).

### Quickstart

```bash
cd experiments/data-quality/frame-level
make install                  # once per checkout
uv run dvc repro              # ensures predictions.json + raw datasets exist

make review-app               # starts the app on http://localhost:8000

# (Review samples in the browser — see "How to use the app" below.)

# End of session — emit the patch + push:
make review-export
uv run dvc add data/09_review data/10_export && uv run dvc push
git add data/09_review.dvc data/10_export.dvc data/.gitignore
git commit -m "review: bbox corrections + export"
```

### Data flow & files persisted

```
data/01_raw/datasets/<split>/labels/<stem>.txt   ← original GT (read-only)
data/01_raw/datasets/<split>/images/<stem>.jpg   ← source images (read-only)
data/07_model_output/<m>/<s>/predictions.json    ← YOLO predictions (read-only)
        │
        ▼
   ┌────────────────────┐
   │  review app (web)  │  reads all three; canvas shows blue=original GT,
   └────────────────────┘  red dashed=predictions, green=corrected GT (editable)
        │
        ▼ (auto-save on every edit)
data/09_review/<m>/<s>/review.json               ← reviewer's corrections
        │
        ▼ (make review-export)
data/10_export/<m>/<s>/
  ├─ labels/<stem>.txt        ← only-changed corrected labels
  ├─ manifest.json            ← apply contract (consumed by pyro-dataset)
  ├─ pending.json             ← unclear-status frames (second-opinion queue)
  └─ provenance.json          ← git/threshold/predictions metadata
```

**Key invariant:** the app never modifies `01_raw/.../labels/*.txt`.
Original GT stays on disk untouched; the reviewer's corrected layer
lives entirely in `review.json` keyed by stem.

### How to use the app

1. **Pick a handle** — first launch shows a modal with preset buttons
   (`arthur` / `mateo` / `felix`) plus a free-form input. Click one to
   continue; the handle persists in `localStorage` and is stamped on
   every saved sample. Click the handle in the header to change it
   later.
2. **Pick a context** — the header dropdowns choose model and split
   (`val`, `train`, or `test`). View chips switch FP / FN. Filters are
   collapsed by default; expand to tune `conf` / `IoU` / `review`
   sliders live (queue rebuilds in 200ms).
3. **Review samples** — the queue (left panel) shows flagged frames
   first, with unflagged sequence siblings dimmed for context. Click a
   thumbnail or use ←/→ to walk the timeline (which shows every frame
   in the current sequence).
4. **Edit bboxes** — on the canvas: drag green box corners to resize,
   drag the body to move, double-click empty space to draw a new box,
   click an original blue GT to copy it into the editable green layer.
   The right panel lists all bboxes with `Use as GT` actions.
5. **Set status** — `Space` marks reviewed and advances; `r`/`u` set
   reviewed/unclear without moving. Auto-save fires 1s after the last
   edit; explicit `✓ saved at <time>` indicator at the bottom right.
6. **See keyboard shortcuts and color legend** — click the `?` icon in
   the header (or press `?`) to toggle the reference panel.

### Sequential hand-off across reviewers

`review.json` is a single per-`(model, split)` file shared via DVC:

```bash
# Mateo, on his machine:
make review-app   # reviews 50 samples, auto-saved to local review.json
uv run dvc add data/09_review && uv run dvc push
git add data/09_review.dvc && git commit -m "review: mateo's val pass" && git push

# Felix, on his machine:
git pull
uv run dvc pull data/09_review   # fetches Mateo's reviewed samples
make review-app                   # Mateo's reviews show as green dots; Felix continues
```

Felix's saves are unioned into the same `review.json` (each sample
carries its own `reviewer` field). The header shows a yellow banner if
the local `review.json` md5 differs from the DVC-tracked md5 — your
cue to `dvc pull` before reviewing to avoid overwriting work.

### Apply the export to pyro-dataset

The export's `manifest.json` is the contract; `pyro-dataset` provides
the apply script (out of scope for this experiment). Rough usage:

```bash
cd ../pyro-dataset
uv run python scripts/apply_audit.py \
    /abs/path/to/vision-rd/experiments/data-quality/frame-level/data/10_export/yolo11s-nimble-narwhal/val/
# Then dvc repro merge_yolo_dataset to propagate to yolo_train_val/yolo_test.
```

`manifest.json::contributors` lists the reviewer handles whose work is
in the patch — useful for crediting in the resulting PR description.

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
     <new-model-name>:
       hf_repo: <hf-org/new-model-repo>
       hf_filename: <pt-filename-in-repo>
       conf_thresh: 0.05
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
  any label it memorized — including incorrect ones — so flags on train
  understate the true label-error rate. Val/test findings are more
  trustworthy. See §10 of the design spec for details.
- **Narwhal v6.0.0 runs at `conf=0.05` per-detection and a temporal
  smoothing threshold of `0.35` in production.** Here there's no
  temporal layer, so we use `conf=0.05` at inference (retain everything)
  and apply a confidence floor in the review app instead, which
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
  09_review/<model-name>/<split>/
    review.json                                     # bbox corrections (DVC-tracked)
  10_export/<model-name>/<split>/
    labels/, manifest.json, pending.json, provenance.json
```

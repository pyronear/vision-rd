# Frame-level review app — design

**Status:** draft
**Date:** 2026-05-05
**Owner:** Arthur

## 1. Background

The frame-level label audit ([`2026-04-24-frame-level-label-audit-design.md`](2026-04-24-frame-level-label-audit-design.md))
runs a YOLO oracle over `pyro-dataset` splits, compares predictions
against the YOLO `.txt` ground truth, and surfaces FP / FN frames in
FiftyOne for human review. Reviewers tag each sample (`label:add-smoke`,
`label:fix-bbox`, …) and tags persist to `data/09_review/<model>/<split>/tags.json`.

The current workflow has two limitations:

1. **No bbox editing.** FiftyOne's tag vocabulary is the only output;
   there is no place to record the *corrected* bbox geometry. The
   downstream pyro-dataset patch still has to be drawn by hand.
2. **Tag taxonomy is a workaround.** `label:add-smoke` / `label:remove-gt`
   / `label:fix-bbox` exist only because tags are the most expressive
   thing FiftyOne lets us record per-sample. With a bbox editor, the
   corrected bbox list speaks for itself and the vocabulary collapses.

This spec describes a self-hosted single-page review app that owns its
own frontend, lets reviewers edit GT bboxes directly, and emits a clean
patch suitable for a PR against `pyro-dataset`.

## 2. Goals

- One reviewer at a time can step through FP/FN samples for a
  `(model, split)` pair, edit GT bboxes inline, and have edits
  auto-persist to disk.
- Threshold parameters (`conf_thresh`, `iou_thresh`, `review_conf_thresh`)
  are live-tunable from the UI; defaults match `params.yaml`.
- Sample queue surfaces sequence groupings naturally so neighboring
  frames are reviewed together.
- Export produces a directory of corrected YOLO `.txt` files plus a
  manifest, ready to drop into `pyro-dataset`.
- Coexists with the existing FiftyOne workflow; FiftyOne is **not**
  removed in this iteration.

## 3. Non-goals

- Multi-reviewer concurrency / shared deployment. Single-reviewer-per-
  checkout, mirrors today's FiftyOne flow.
- Auth, TLS, multi-tenant deploy.
- Multi-model overlay (multiple models on the same image at once). The
  app shows one model at a time; switch models via dropdown.
- Migrating existing `tags.json` data. The new app starts fresh from
  `review.json`. The two files coexist.
- Modifying the existing DVC pipeline (`predict_*`, `build_fiftyone_*`).
  The app reads `predictions.json` lazily.

## 4. User flow

```
make install                          # one-time, inside experiment dir
uv run dvc repro                      # ensures predictions.json exists
make review-pull                      # fetch existing review.json from DVC (skip on first run)
make review-app                       # starts uvicorn on localhost:8000
                                      #   → opens app in browser

(reviewer picks model + split from header dropdowns;
 walks queue with arrow keys; edits bboxes with mouse;
 sets status; auto-saves to review.json)

make review-export                    # build YOLO .txt patch + manifest under data/10_export/
uv run dvc add data/09_review data/10_export
uv run dvc push
git add … && git commit
```

`make review-app` and `make review-export` are new targets. Existing
`make fiftyone-fp` / `make fiftyone-fn` continue to work unchanged.

## 5. Data model

### 5.1 Review state — `data/09_review/<model>/<split>/review.json`

```json
{
  "version": 1,
  "model_name": "yolo11s-nimble-narwhal",
  "split": "val",
  "samples": {
    "<image_stem>": {
      "status": "reviewed" | "unclear",
      "bboxes": [
        {"class_id": 0, "cx": 0.28, "cy": 0.53, "w": 0.013, "h": 0.025}
      ],
      "reviewer": "arthur",
      "note": "moved bbox up; pred was on cloud, removed FP",
      "reviewed_at": "2026-05-05T14:21:33Z"
    }
  }
}
```

Properties:

- **Untouched samples are absent.** The file only carries decisions
  the reviewer made.
- **`bboxes` is the canonical corrected GT — additive, not destructive.**
  The original GT (from `data/01_raw/datasets/<split>/labels/<stem>.txt`)
  and the model predictions (from `predictions.json`) are **never
  modified**. They remain on disk untouched and are rendered as
  read-only reference layers in the UI on every load (see §6.4). The
  `bboxes` list is a separate "corrected GT" layer the reviewer
  authors. Empty list = "remove all GT here for export purposes."
  Non-empty = "this is what GT should be for export." The original
  `.txt` stays exactly as it was so the next reviewer can audit the
  decision against the source material.
- **Export uses `bboxes` only.** §5.2's exported `.txt` files contain
  the corrected bboxes, not a union with the original. The original
  stays on disk for traceability, but the patch the export emits is
  the reviewer's final answer.
- **`status`** carries only the two states that are not expressible as
  bbox edits: `unclear` (skip / second opinion) and `reviewed`
  (decision made, even if "GT was already correct" — in that case
  `bboxes` is a copy of the original GT).
- **`reviewer`** and **`note`** are optional.
- **`reviewed_at`** is ISO-8601 UTC, set by the server on each save.

Stems in `samples` are written sorted; `bboxes` lists are written
in canvas order (top-to-bottom, left-to-right) for stable diffs.

This file replaces `tags.json`. The old `tags.json` files stay on disk
and remain readable by the FiftyOne workflow; the new app does not
read them.

### 5.2 Export — `data/10_export/<model>/<split>/`

```
labels/<stem>.txt                # only stems whose corrected bboxes
                                 # differ from the original .txt
manifest.json                    # which stems changed and a summary diff
```

`labels/<stem>.txt` is a YOLO-format file (`class cx cy w h`, one box
per line) carrying the **corrected** bboxes — flat (no split subdir),
one file per changed stem.

**Apply target — upstream, not merged outputs.**
`pyro-dataset`'s `dvc.yaml` regenerates `processed/yolo_train_val/` and
`processed/yolo_test/` on each `dvc repro` by `merge_yolo_dataset.py`,
which `shutil.copy2`s files from upstream `processed/wildfire_yolo/` and
`processed/fp_yolo/` (both flat, no split subdir). Patches dropped into
`yolo_train_val/labels/<split>/` would be overwritten on the next
`dvc repro`. The correct apply target is whichever upstream the stem
came from — `wildfire_yolo/labels/<stem>.txt` or
`fp_yolo/labels/<stem>.txt` — followed by a `dvc repro` to propagate.

Stems are unique across the two upstreams (a given stem appears in
exactly one of `wildfire_yolo` or `fp_yolo`), so the apply step can
route each patch by checking which upstream contains the stem. The
apply step itself is **out of scope for this spec** (lives on the
`pyro-dataset` side as a separate script consuming our `manifest.json`).
Our export only emits the corrected `.txt` files plus the manifest.

`manifest.json`:

```json
{
  "version": 1,
  "model_name": "yolo11s-nimble-narwhal",
  "split": "val",
  "exported_at": "2026-05-05T14:30:00Z",
  "changed": [
    {
      "stem": "hpwren-figlib_rmwmoboc_999_2018-07-29T00-19-06",
      "added": 1, "removed": 0, "modified": 0,
      "reviewer": "arthur",
      "note": "..."
    }
  ],
  "totals": {"changed": 17, "added": 8, "removed": 4, "modified": 9}
}
```

`changed` is sorted by stem. Counts use a simple greedy match: for each
original bbox, find the closest corrected bbox by IoU; pairs with IoU ≥
0.95 count as unchanged, pairs with IoU < 0.95 count as `modified`,
unmatched originals are `removed`, unmatched corrected boxes are `added`.

`unclear` samples are **excluded** from the export. They are not
decisions; they are open questions.

## 6. UI

Single-page workbench layout, three columns + header:

```
┌─────────────────────────────────────────────────────────────────────┐
│ HEADER: model ▾  split ▾  view: [FP|FN|All]  reviewer · 47/113 done │
├──────────┬───────────────────────────────────┬──────────────────────┤
│ FILTERS  │                                   │ BBOXES (this frame)  │
│ ▸ conf   │                                   │ ▸ GT #0 · TP   ✏ ✕   │
│ ▸ IoU    │           IMAGE +                 │ ▸ pred · FP    →GT   │
│ ▸ revC   │           BBOX EDITOR             │ [+ add GT box]       │
│ FP/FN/   │                                   │                      │
│ progress │                                   │ STATUS               │
│          │                                   │ [reviewed][unclear]  │
│ QUEUE    ├───────────────────────────────────┤                      │
│ (sorted, │                                   │ NOTE                 │
│ grouped  │     TIMELINE STRIP (sequence)     │ [textarea]           │
│ by seq)  │                                   │                      │
│          │                                   │ ✓ auto-saved 2s ago  │
└──────────┴───────────────────────────────────┴──────────────────────┘
```

### 6.1 Header

- **Model dropdown** — populated from `params.yaml`'s `models:` keys.
- **Split dropdown** — `train` / `val` / `test`.
- **View chip** — `FP` / `FN` / `All` (pick one). Default: `FP`.
- **Reviewer name** — free-text input, persisted to localStorage and
  written into each `review.json` save as the `reviewer` field.
- **Progress** — `<reviewed-count> / <queue-length>` for the current
  view + filters.

Switching model or split reloads the queue. The reviewer's editing
state on the current sample is auto-saved before reload.

### 6.2 Filter panel (left)

Three sliders:

| Slider              | Range          | Default | Effect                                                         |
|---------------------|----------------|---------|----------------------------------------------------------------|
| `conf ≥`            | 0.05 – 1.0     | 0.05    | Hide predictions below this confidence everywhere.             |
| `IoU ≥`             | 0.0 – 1.0      | 0.05    | Threshold used to assign TP / FP / FN per frame.               |
| `review conf ≥`     | 0.0 – 1.0      | 0.35    | Floor applied to the FP queue (mirror of FiftyOne's filter).   |

`conf ≥` cannot go below 0.05 — that is the inference-time floor used
by `predict_*`; lower values are not in `predictions.json`.

All three sliders **live-recompute** TP/FP/FN matching server-side and
push a fresh queue to the client. With ~14k frames and tens of
predictions per frame, full recompute is milliseconds in Python.

When the queue length changes due to a slider, the app keeps the
reviewer on their current sample if it is still in the queue;
otherwise it advances to the next sample in the new queue's natural
sort order (or, if the current sample was past the end, the last
queue entry).

A FP / FN / All chip row underneath the sliders selects the queue
contents.

### 6.3 Queue (left, below filters)

A scrollable list of items. **Sort key depends on the active view:**

- `FP` view: `(sequence_max_fp_confidence DESC, sequence_id, timestamp ASC)`.
  Mirrors FiftyOne's `fp-by-confidence` — hottest sequence first.
- `FN` view: `(sequence_max_fn_area DESC, sequence_id, timestamp ASC)`.
  Mirrors FiftyOne's `fn-by-area` — largest missed annotation first.
- `All` view: `(sequence_max_severity DESC, sequence_id, timestamp ASC)`,
  where severity = max(fp_conf, fn_area) normalized to [0, 1].

In every case the secondary keys cluster sibling frames together, so
adjacent items in the queue are also adjacent in time within a sequence
— no explicit grouping toggle needed.

Each sequence boundary in the queue gets a small sticky header showing
`📷 <camera>_<seq_id> · <date>` and the count of flagged frames in that
sequence. The header is for orientation only — clicking it does
**not** filter the queue.

Each item shows: 36×36 thumbnail with mini-bbox, stem suffix
(timestamp portion only — the prefix is implied by the sequence
header), kind (`FP` / `FN`) and confidence, and a status dot
(green=reviewed, yellow=unclear, gray=pending).

### 6.4 Image + bbox editor (center)

The canvas renders **three layers** simultaneously, distinguished by
color/style. Two of the three are always read-only reference; only one
is editable.

| Layer            | Source                          | Style                              | Editable |
|------------------|---------------------------------|------------------------------------|----------|
| Original GT      | `01_raw/<split>/labels/<stem>.txt` | solid blue, semi-transparent fill, no handles | no |
| Predictions      | `07_model_output/.../predictions.json` | dashed red, no fill, no handles | no |
| **Corrected GT** | `09_review/<model>/<split>/review.json::samples[<stem>].bboxes` | solid green, full opacity, corner handles | **yes** |

Visual rules:

- Labels above each box: `GT (original) #0 · TP`, `pred · FP · 0.58`,
  `GT (corrected) #0`.
- An original GT that is **not** present in the corrected list (the
  reviewer effectively removed it) is rendered with a diagonal strike
  pattern and labeled `GT (original) · removed`.
- An original GT that **is** present in the corrected list with
  identical geometry is shown only in green (corrected); the original
  is suppressed to keep the canvas clean.
- An original GT that the reviewer moved/resized appears as **two
  boxes**: the original (blue, semi-transparent, no handles, labeled
  `GT (original)`) and the corrected version (green, with handles,
  labeled `GT (corrected)`). A faint connector line joins them so the
  override relationship is visible.
- A new corrected GT (no original counterpart) appears in green only,
  labeled `GT (corrected) · added`.
- Layer toggles (top-right of canvas): `[O] original GT` and
  `[P] predictions`. Both default on; reviewer can hide either to
  declutter while editing. The corrected layer is always visible.

Editing affordances (corrected layer only):

- **Double-click empty space** starts a new corrected GT box (drag to
  set extents, release to commit).
- **Click an original GT** copies it into the corrected list at
  identical geometry; subsequent drag/resize edits the corrected copy.
  This is the "I'm fixing this box" interaction — the original stays
  blue underneath as the reviewer's reference.
- **Drag corner / body** of a corrected box resizes / moves it.
- **Selected corrected box** can be deleted with the Delete key.
- **Delete key on a selected original GT** marks it removed (drops it
  from the corrected list if it was there; if it wasn't there, the
  reviewer's "remove" intent is recorded by simply not adding it).
  Visual feedback: the original switches to the struck-through style.

The canvas is HTML5 `<canvas>`; coordinates round-trip through the
YOLO normalized `(cx, cy, w, h)` representation so what is saved is
exactly what is drawn.

### 6.5 Timeline strip (center, below image)

A horizontal filmstrip of frames in the **current sample's sequence**,
windowed around the current frame (5 before, 5 after by default;
horizontally scrollable for longer sequences). For each frame:

- 64×36 thumbnail showing the frame's bboxes as miniatures.
- Timestamp suffix below the thumbnail.
- Status dots: red for FP, yellow for FN, green for reviewed.
- The current frame has a blue border.

Clicking a thumbnail navigates to that frame even if it is not in the
queue (sequence-level navigation).

### 6.6 Right panel — bbox list, status, note

- **Bbox list** — three sections matching the canvas layers:
  1. **Original GT** rows (blue) — read-only, show source + normalized
     coords + TP/FP/FN status. Each row has a `Use as GT` action that
     copies the original into the corrected list (same as clicking the
     box on canvas).
  2. **Corrected GT** rows (green) — edit / delete affordances. Rows
     that override an original show a small "→ #N" indicator linking
     them to the original they correct.
  3. **Predictions** rows (red) — read-only, show confidence and
     TP/FP/FN status. Each row has a one-click `Use as GT` action that
     promotes the prediction's geometry to a new corrected GT box (the
     former `label:add-smoke` workflow).
- **Add GT box** button — same as double-clicking empty canvas;
  appends to the corrected list.
- **Status** — radio-style two-button group: `reviewed` / `unclear`.
  Default is `reviewed` once any edit happens.
- **Note** — free-text textarea, optional.
- **Save indicator** — green bar at the bottom showing
  `✓ auto-saved <N>s ago`. Turns to a red `unsaved` indicator while a
  save is pending.

### 6.7 Keyboard shortcuts

- **← / →** step through sibling frames in the current sequence
  (timeline-strip order; flagged or not).
- **Ctrl ← / Ctrl →** jump to the first flagged frame of the previous
  / next sequence in the queue.
- **Delete** delete selected bbox (corrected: removes from list;
  original: marks as removed for export).
- **Esc** clear bbox selection.
- **u** toggle status to `unclear`.
- **r** toggle status to `reviewed`.
- **o** toggle visibility of the original GT layer.
- **p** toggle visibility of the predictions layer.

## 7. Persistence

### 7.1 Auto-save

Every meaningful state change (bbox edited / added / deleted, status
changed, note edited) marks the sample dirty. A debounced timer
(1 second after the last change) triggers a save.

Navigating away from a sample (←, →, Ctrl ←, Ctrl →, click in queue,
click in timeline, dropdown change) flushes the dirty state immediately
before navigating.

### 7.2 File-write atomicity

Saves are **whole-file writes** of `review.json`: the server holds the
in-memory map for the active `(model, split)`, applies the per-sample
update, serializes the entire payload, writes to a sibling
`review.json.tmp`, fsync, then `rename(tmp, review.json)`. This makes
the save atomic against crashes. The file is small (≤ 113 entries on
val today) so full rewrites are cheap.

### 7.3 Reload behavior

On startup and on `(model, split)` change, the server reads
`review.json` if it exists and seeds the in-memory map. If
`review.json` is missing, the server starts with an empty map and will
create the file on the first save.

## 8. Threshold recomputation

The server keeps an in-memory snapshot of `predictions.json` and the
parsed GT `.txt` files for the active `(model, split)`. On each slider
change, the server recomputes per-frame TP/FP/FN matching with the new
thresholds and the new queue, and pushes the result to the client over
WebSocket (or simple polling for the first iteration; details in the
implementation plan).

Matching is the same algorithm FiftyOne's `evaluate_detections` uses:
greedy IoU-based assignment per frame, predictions and GT both scored
against the IoU floor. We re-implement it here (not via FiftyOne) so
we have no FiftyOne dependency in the app.

## 9. Tech stack

- **Backend:** FastAPI + uvicorn, run via `uv run`. Pure Python; no
  Mongo, no FiftyOne dependency. Reads `predictions.json` and the GT
  `.txt` tree directly.
- **Frontend:** single static HTML page + vanilla JS + a single
  `<canvas>` for the bbox editor. No bundler, no React, no build step.
  CSS in one file.
- **Image serving:** the FastAPI app exposes `/image/<stem>` that
  reads from `data/01_raw/datasets/<split>/images/<stem>.jpg`.

The app lives at:

```
experiments/data-quality/frame-level/
  src/data_quality_frame_level/review_app/
    __init__.py
    main.py            # FastAPI app factory, route registration
    state.py           # in-memory model: predictions, GT, review state
    matching.py        # TP/FP/FN matching against thresholds
    persistence.py     # review.json read/write (atomic)
    export.py          # YOLO .txt + manifest emitter
    static/
      index.html
      app.js
      app.css
  scripts/
    run_review_app.py    # uvicorn entrypoint with CLI args
    export_review_app.py # CLI wrapper around export.py
```

`Makefile` adds:

```
review-app:
\tuv run --group review-app python scripts/run_review_app.py

review-export:
\tuv run python scripts/export_review_app.py
```

`pyproject.toml` adds an optional dependency group `review-app`
(`fastapi`, `uvicorn`) so the existing pipeline does not pull them in.

## 10. Coexistence with FiftyOne

- The new app reads/writes **only** `review.json`.
- The FiftyOne workflow continues to read/write **only** `tags.json`.
- They share neither file. A reviewer who opens both will see two
  separate review states; this is acceptable for the transition
  period.
- No FiftyOne code paths or `make` targets are removed in this
  iteration.

A migration script (out of scope for this spec) can later replay
`tags.json` decisions into `review.json` if needed — most decisions
(`label:ok`, `status:unclear`) map cleanly to `status: reviewed` /
`status: unclear` with no bbox edits.

## 11. Out of scope

- Multi-reviewer concurrency, locking, conflict resolution.
- Authentication, TLS, deployment beyond `localhost`.
- Bulk operations (e.g. "mark every frame in this sequence reviewed").
- A migration that reads the old `tags.json` into `review.json`.
- Edit history / per-edit attribution. The latest reviewer's
  `bboxes` overwrites whatever was there; the original on-disk `.txt`
  is the only "before" reference. Audit happens via git/DVC history of
  `review.json`.

## 12. Open questions

- **Bbox class.** Production has `class_id = 0` (smoke). The editor
  defaults new boxes to class 0. If multi-class arrives, the bbox row
  will need a class selector. For now: hard-coded.
- **Sequence detection from stems.** Validated against the live
  `data/01_raw/datasets/{train,val,test}/labels/` tree (18,833 files):
  every stem has exactly four `_`-separated fields:
  `<source>_<camera>_<sequence_id>_<timestamp>`. Sources may contain
  hyphens (`awf-axis`, `pyronear-force-06`) but never underscores.
  `rsplit('_', 1)` reliably yields `(prefix, timestamp)` for every
  stem in every split. No edge cases.

  A single prefix can recur across many disjoint detection events
  spanning days or weeks (same camera + dataset `sequence_id`). To
  avoid treating those as one sequence, the app additionally splits
  on **temporal gaps**: two stems share a temporal sequence iff they
  share a prefix AND no gap larger than 180 seconds separates two
  consecutive frames in their cluster. Successive clusters under the
  same prefix become `prefix#0`, `prefix#1`, … (chronological).

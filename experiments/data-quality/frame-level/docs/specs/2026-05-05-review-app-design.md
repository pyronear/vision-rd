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
- **`bboxes` is the canonical corrected GT.** Empty list = "remove all
  GT here". Non-empty = "this is what GT should be." The original
  `.txt` is always recoverable from
  `data/01_raw/datasets/<split>/labels/<stem>.txt`, so the diff is
  computable on demand.
- **`status`** carries only the two states that are not expressible as
  bbox edits: `unclear` (skip / second opinion) and `reviewed`
  (decision made, even if "GT was already correct").
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
per line) carrying the **corrected** bboxes. It mirrors the layout of
`pyro-dataset`'s `processed/yolo_train_val/labels/<split>/<stem>.txt`
so applying the patch is a copy.

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

- Image fills the available space, letterboxed.
- **GT bboxes** drawn in solid blue, with corner handles for resize and
  body-drag for move. **Predictions** drawn in dashed red, read-only.
  Labels above each box show source + status (`GT #0 · TP`,
  `pred · FP · 0.58`).
- **Double-click empty space** starts a new GT box (drag to set
  extents, release to commit).
- **Selected bbox** can be deleted with the Delete key.
- The canvas is HTML5 `<canvas>`; coordinates round-trip through the
  YOLO normalized `(cx, cy, w, h)` representation so what is saved is
  exactly what is drawn.

The editor never modifies predictions. Predictions stay as the model
emitted them; they are reference geometry only.

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

- **Bbox list** — all GT bboxes for the current frame plus all
  predictions ≥ `conf` slider. Each row shows source, normalized
  coords, TP/FP/FN status; predictions get a one-click `→GT` action
  that promotes the prediction's geometry to a new GT box (the
  former `label:add-smoke` workflow). GT rows have edit + delete
  affordances.
- **Add GT box** button — same as double-clicking empty canvas.
- **Status** — radio-style two-button group: `reviewed` / `unclear`.
  Default is `reviewed` once any edit happens.
- **Note** — free-text textarea, optional.
- **Save indicator** — green bar at the bottom showing
  `✓ auto-saved <N>s ago`. Turns to a red `unsaved` indicator while a
  save is pending.

### 6.7 Keyboard shortcuts

- **← / →** step through global queue.
- **Ctrl ← / Ctrl →** step through sibling frames in the current
  sequence (queue-irrelevant; flagged or not).
- **Delete** delete selected bbox.
- **Esc** clear bbox selection.
- **u** toggle status to `unclear`.
- **r** toggle status to `reviewed`.

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
- Diff-vs-original visual mode (showing original GT alongside corrected
  GT). Useful, but additive; can ship later.
- A migration that reads the old `tags.json` into `review.json`.

## 12. Open questions

- **Bbox class.** Production has `class_id = 0` (smoke). The editor
  defaults new boxes to class 0. If multi-class arrives, the bbox row
  will need a class selector. For now: hard-coded.
- **Sequence detection from stems.** Stems look like
  `<source>_<camera>_<sequence_id>_<timestamp>`. We split on the last
  `_` to extract the timestamp; everything before is the sequence id.
  This needs a small unit test against pyro-dataset's actual stems —
  there are some that do not perfectly match the pattern (e.g.
  `hpwren-figlib_*_2019-07-16T00-18-24` — `2019-07-16T00-18-24` is the
  timestamp, the rest is sequence id). The implementation plan should
  validate this on the real `01_raw` tree.

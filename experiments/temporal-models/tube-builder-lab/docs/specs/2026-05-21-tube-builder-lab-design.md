# Tube Builder Lab — design

**Date:** 2026-05-21
**Status:** approved (brainstorm)
**Scope:** this PR builds the lab and iterates on a candidate tube-linking
algorithm against real failure cases. Propagating the winner to
`lib/bbox-tube-temporal` and retraining are explicitly a **later PR**.

## Problem

The smoke-tube builder (`lib/bbox-tube-temporal/.../tubes.py::build_tubes`)
over-fragments: it produces many tubes where there should be one. Reviewing
the temporal model in the explorer, we collected 16 sequences where two or
more tubes should be a single tube.

Root cause hypothesis: greedy one-to-one IoU matching with `iou_threshold=0.2`
and `max_misses=2` over frames that are ~30s apart. A smoke plume grows and
drifts between captures, so the IoU between consecutive detections drops below
threshold (the early small box is *contained* in the later large box rather
than overlapping it heavily), and detector gaps longer than two frames
terminate a tube that is later re-detected as a fresh one. The collected cases
show two failure shapes:

- **Adjacent splits** — consecutive tubes that should be continuous
  (e.g. `41304` three→one, `41310` T3≈T4). Better association (IoU **or**
  containment) targets these.
- **Distant re-detections** — a plume lost then re-detected many frames later
  as a new tube (e.g. `43206` T0≈T5, `41289` T0≈T3). Gap bridging or a
  post-hoc merge pass targets these.

We will not commit to a fixed set of techniques up front. The lab is an
**example-driven comparison harness**: we edit a candidate builder freely and
visually confirm it fixes each real case without regressing good ones.

## Goals

- See, side by side, what the **current** builder and a **candidate** builder
  produce on a given sequence — fast enough to iterate in seconds.
- Drive iteration from the 16 collected failure cases (the working set), with
  their notes visible.
- Catch regressions: a candidate that newly over-merges an already-correct
  sequence must be obvious at a glance.

## Non-goals (this PR)

- Changing `lib/bbox-tube-temporal` (the candidate lives in the experiment).
- Retraining the temporal model.
- Reproducing the keep/discard decision or running the classifier. We judge
  **tube structure** only.

## Working set

Stored in `working_set.yaml`. `targets` are the collected failure cases;
`control` is a small set of already-correct sequences we watch for regressions
(picked during implementation from clean single-plume sequences).

```yaml
targets:
  - { key: platform_43096 }
  - { key: platform_42466 }
  - { key: platform_41304, note: "the three tubes should only be one" }
  - { key: platform_41319 }
  - { key: platform_41310, note: "T3 and T4 should be the same tube?" }
  - { key: platform_41289, note: "T0 and T3 should be the same tube" }
  - { key: platform_41209 }
  - { key: platform_40800 }
  - { key: platform_41616, note: "the three tubes should only be one" }
  - { key: platform_41786, note: "the tubes could be merged into one" }
  - { key: platform_42562 }
  - { key: platform_42538, note: "should be only one tube" }
  - { key: platform_41887 }
  - { key: platform_41541 }
  - { key: platform_43206, note: "T0 and T5 should be one; T1 and T3 should be one" }
  - { key: platform_42910 }
control: []   # filled during implementation
```

## Architecture

A new standalone uv experiment, `experiments/temporal-models/tube-builder-lab/`
(from `experiments/template/`). **Fully isolated**: its own env, its own
DVC-tracked data, no dependency on `temporal-model-explorer`. It depends only on
`bbox_tube_temporal` (current builder, types, `run_yolo_on_frames`) and
`pyrocore` via `[tool.uv.sources]`.

Mirrors the explorer's split of offline compute vs. read-only viewer:

```
platform API ──import_sequences.py──> data/03_primary/sequences/platform_<id>/  (frames + minimal meta)  [DVC]
model package ─────────────────────> data/06_models/<name>/model.zip            (one model; for its YOLO)  [DVC]
sequences + model ──cache_detections.py──> data/05_model_input/detections/<key>.json  (run YOLO ONCE)     [DVC]
                                              │
detections + working_set.yaml ──> app.py ──> current tubes  (lib build_tubes + filter)
                                          └──> candidate tubes (candidate.py, hot-reloaded)
                                                  │
                                          Layout A viewer (Streamlit)
```

### Components

**`scripts/import_sequences.py`** (offline, needs platform creds)
Imports the working-set sequences **by id** into the experiment's own store.
Duplicates a lean platform client (~60 lines, mirror of the explorer's
`platform_api.py`) — accepted duplication, in exchange for full isolation.
Flat `data/03_primary/sequences/platform_<id>/` layout (we navigate by
`working_set.yaml`, not by org/camera). Writes a minimal `meta.json` (key,
sequence_id, frames ordered by `created_at`); label/camera are optional display
extras. Output DVC-tracked and pushed.

**`scripts/cache_detections.py`** (offline, the only slow/GPU step)
Loads a model package (`--model-zip`, default `data/06_models/<name>/model.zip`)
to reuse its **exact** bundled YOLO weights + detection params (conf, NMS,
imgsz), so detections match what `BboxTubeTemporalModel.predict` produces. Runs
`run_yolo_on_frames` over each sequence, serializes per-frame detections
(`frame_idx, frame_id, cx, cy, w, h, confidence, class_id`) to
`data/05_model_input/detections/<key>.json`. Idempotent (skips cached). Output
DVC-tracked. Optionally wired as a `dvc.yaml` stage (deps: sequences +
model.zip; outs: detections).

**`src/tube_builder_lab/candidate.py`** — the heart of the loop
A single function `build_tubes_candidate(frame_detections: list[FrameDetections])
-> list[Tube]`. Seeded as a copy of the lib's current `build_tubes`, so
**current == candidate on day one** (empty diff). We edit it freely against the
examples — containment/IoMin association, larger `max_misses`, a post-hoc merge
pass, anything a case demands. Builds on the lib's primitives (`compute_iou`,
types) so the eventual propagation is mechanical. The app hot-reloads this
module (`importlib.reload`) on a **Re-run candidate** button — edit the file,
see the new tubes without restarting.

**`src/tube_builder_lab/pipeline.py`** — comparable two-sides
`detections_to_display_tubes(frame_detections, builder, *, truncate)` applies:
optional truncation (to the model's `max_frames`), the chosen builder, then the
lib's `filter_and_interpolate_tubes` with the model's params. "current" passes
the lib `build_tubes`; "candidate" passes `build_tubes_candidate`. Both sides
use the **same** truncation setting so the A/B stays fair.

**`src/tube_builder_lab/viz.py`** — pure, unit-tested
Adapts the explorer's helpers to take `Tube` objects directly (not the
details-JSON shape): the Altair tube timeline, `draw_bboxes` (bboxes colored by
tube id), and `crop_around_bbox`. Copied into this experiment per monorepo
isolation.

**`app.py`** — Streamlit viewer (Layout A)
Reads only the DVC-tracked data; never imports models or calls the API.
Per selected sequence:
- **Working-set navigator** — ◀ prev / next ▶ through the targets (and
  control), each showing its note.
- **Shared frame player** — autoplaying (pausable), bboxes colored by tube id,
  with a current ⇄ candidate coloring toggle.
- **Two stacked tube timelines** on the same frame axis: current (top) above
  candidate (bottom). Metric `current N → candidate M` tubes.
- **Candidate crop strip** — the per-tube context crops, secondary/collapsible,
  to sanity-check a merge spatially.
- **Truncation toggle** — ON by default (reproduces the explorer 1:1); off
  studies the full untruncated trajectory. Shared across both sides.
- **Re-run candidate** button — hot-reloads `candidate.py`.
- **Working-set summary table** — every sequence's `current → candidate` tube
  count, so a regression (good sequence that newly over-merges) stands out.

## Success criteria

- On each of the 16 targets, the candidate produces the merges noted in
  `working_set.yaml` — confirmed visually in Layout A and by the tube-count
  drop in the summary table.
- The control set shows no new over-merging (counts stay correct).
- Iteration latency after caching is seconds (edit `candidate.py` → Re-run).

## Testing

- `viz.py`: pure helpers (timeline dataframe shaping, bbox pixel mapping) unit
  tested.
- `pipeline.py`: truncation + filter wiring tested with synthetic detections.
- `cache_detections` serialization round-trips (Detection ↔ JSON).
- `import_sequences`: meta/frame ordering tested with a stubbed downloader
  (mirrors the explorer's test approach).
- `candidate.py`: behavior tests grow case-by-case as we encode each fix.

## Open questions

None blocking. If a third consumer of the platform client appears, promote it
to `lib/` rather than duplicating a third time.

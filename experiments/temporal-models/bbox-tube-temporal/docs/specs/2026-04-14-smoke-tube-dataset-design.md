# Smoke Tube Dataset — Design

**Date:** 2026-04-14
**Status:** Approved, ready for implementation plan
**Scope:** `experiments/temporal-models/smokeynet-adapted/`

## Goal

Build a reproducible pipeline that turns the existing sequence-level
WF/FP dataset into a **per-sequence smoke-tube dataset** suitable for
training. The pipeline reads label `.txt` files directly (no YOLO
pass), produces **one tube per sequence** with a binary
`smoke`/`non-smoke` label, and stores metadata only (no features or
cropped patches).

**Non-goals (v1):** multi-tube-per-sequence, per-entry supervision
labels, feature extraction, augmentation, online inference path. Those
are deferred to follow-up work.

## Inputs

Existing Kedro-style raw dataset:

```
data/01_raw/datasets_full/{train,val}/
  wildfire/<sequence_id>/
    images/*.jpg
    labels/*.txt        # 5-col: class cx cy w h   (human GT)
  fp/<sequence_id>/
    images/*.jpg
    labels/*.txt        # 6-col: class cx cy w h conf  (YOLO predictions that fired)
```

Key property: the label files already carry everything needed to build
tubes. WF frames carry clean GT chains; FP frames carry the exact YOLO
predictions that triggered false positives in production. No YOLO
inference is required.

## Outputs

```
data/03_primary/tubes/{train,val}/
  <sequence_id>.json    # one tube per surviving sequence
  _summary.json         # per-split stats + dropped sequences with reasons
```

### `<sequence_id>.json` schema

```json
{
  "sequence_id": "adf_avinyonet_999_2023-05-23T17-18-31",
  "split": "train",
  "label": "smoke",
  "source": "gt",
  "num_frames": 114,
  "tube": {
    "start_frame": 0,
    "end_frame": 113,
    "entries": [
      {
        "frame_idx": 0,
        "frame_id": "adf_avinyonet_999_2023-05-23T17-18-31",
        "bbox": [0.4436, 0.3296, 0.0240, 0.0366],
        "is_gap": false,
        "confidence": 1.0
      }
    ]
  }
}
```

Fields:

- `label` is `"smoke"` (from WF) or `"fp"` (from FP).
- `source` is `"gt"` (5-col) or `"yolo"` (6-col) — records provenance of bboxes.
- `bbox` is `[cx, cy, w, h]` normalised in `[0, 1]`.
- `is_gap=true` entries have a geometrically-interpolated bbox and
  `confidence=0.0`.
- `confidence` is `1.0` for GT entries, the stored value for observed
  YOLO entries, `0.0` for gap entries.

### `_summary.json` schema

```json
{
  "split": "train",
  "total_sequences": 640,
  "tubes_written": 512,
  "by_label": {"smoke": 280, "fp": 232},
  "dropped": [
    {"sequence_id": "...", "reason": "no_labels_dir"},
    {"sequence_id": "...", "reason": "no_detections"},
    {"sequence_id": "...", "reason": "too_short"}
  ]
}
```

## Pipeline

```
Per sequence:
  load_frame_detections(seq_dir)
    → parses 5-col (WF) or 6-col (FP) labels into list[FrameDetections]
  build_tubes(frame_dets, iou_threshold, max_misses)        # existing
    → candidate tubes with detection=None in gap entries
  select_longest_tube(tubes)
    → one Tube (tie-break by most detected entries)
  interpolate_gaps(tube)
    → gap entries get lerped bbox, is_gap=True, confidence=0.0
  apply filters
    → min_tube_length, min_detected_entries
  serialize JSON
```

## Tube construction decisions

- **Gap handling:** geometric linear interpolation of `(cx, cy, w, h)`
  at tube-build time. Interior gaps use `lerp(bbox_before, bbox_after, t)`;
  boundary gaps repeat the nearest observed bbox. Every gap entry has
  `is_gap=True` and `confidence=0.0`.
- **Selection:** one tube per sequence, chosen as the tube with the
  most entries; tie-break by number of non-gap (observed) entries.
  Applied identically to WF and FP.
- **WF vs FP source:** WF reads 5-col GT labels, FP reads 6-col YOLO
  predictions. Same `build_tubes` + `select_longest_tube` + `interpolate_gaps`
  code path in both cases — only the loader differs.
- **Accepted limitation:** WF tubes derive from clean GT bboxes while
  FP tubes derive from noisier YOLO bboxes. This bbox-source mismatch
  can create a shortcut for the model ("clean bbox = smoke"). Flagged
  as a known risk to revisit once the v1 pipeline is validated.

## Filters (v1 — simple defaults, tuneable)

- `min_tube_length = 4` — tube spans at least 4 frames (gaps included)
- `min_detected_entries = 2` — at least 2 observed bboxes (non-gap)
- Drop WF sequences with no labels → reason `"no_labels_dir"` or `"no_detections"`
- Drop FP sequences with fewer than `min_detected_entries` total detections

The `_summary.json` output records every drop with its reason so we
can loosen thresholds after seeing the dropout rate.

## Code changes

### `src/smokeynet_adapted/types.py`

Extend `TubeEntry`:

```python
@dataclass
class TubeEntry:
    frame_idx: int
    detection: Detection | None
    is_gap: bool = False
```

Pre-interpolation gaps have `detection=None, is_gap=True`; post-interpolation
gaps have a lerped `Detection` and `is_gap=True`.

### `src/smokeynet_adapted/data.py`

Add:

- `load_detections(sequence_dir, frame_id) -> list[Detection]` — unified
  reader. 5-col → `confidence=1.0`; 6-col → confidence from column 6.
  Malformed lines are warned and skipped.
- `load_frame_detections(sequence_dir) -> list[FrameDetections]` — loops
  over all frames in the sequence, builds ordered `FrameDetections`.

### `src/smokeynet_adapted/tubes.py`

Add:

- `interpolate_gaps(tube: Tube) -> Tube` — in-place or returns a new
  tube; replaces `detection=None` entries with lerped `Detection`
  objects (`confidence=0.0`, `is_gap=True`). Boundary gaps repeat
  nearest observed detection.
- `select_longest_tube(tubes) -> Tube | None` — returns the tube with
  the most entries; tie-break by number of non-gap entries. Returns
  `None` on empty input.

### `scripts/build_tubes.py` (new)

CLI entrypoint:

```
uv run python scripts/build_tubes.py \
  --input-dir  data/01_raw/datasets/<split> \
  --output-dir data/03_primary/tubes/<split> \
  [--iou-threshold 0.2] [--max-misses 2] \
  [--min-tube-length 4] [--min-detected-entries 2]
```

All commands are run via `uv run` (matches the existing `make` /
`dvc.yaml` conventions in this experiment).

Walks `{wildfire,fp}/*/`, runs the pipeline above per sequence, writes
JSON per surviving sequence plus `_summary.json`.

### `notebooks/02-visualize-built-tubes.ipynb` (new)

- Loads one split's tube JSONs from disk.
- Renders with the existing `plot_tube_summary` / `draw_tubes_on_frames`
  visualisations (the same ones used by `01-visualize-smoke-tubes.ipynb`).
- No YOLO model dependency — purely a dataset-inspection tool.

The existing `01-visualize-smoke-tubes.ipynb` stays untouched as the
online-YOLO sanity-check notebook.

### `dvc.yaml`

New `build_tubes` stage (after `truncate`):

```yaml
build_tubes:
  foreach: [train, val]
  do:
    cmd: >-
      uv run python scripts/build_tubes.py
      --input-dir data/01_raw/datasets/${item}
      --output-dir data/03_primary/tubes/${item}
    deps:
      - scripts/build_tubes.py
      - src/smokeynet_adapted/tubes.py
      - src/smokeynet_adapted/data.py
      - src/smokeynet_adapted/types.py
      - data/01_raw/datasets/${item}
    params:
      - tubes
      - build_tubes
    outs:
      - data/03_primary/tubes/${item}
```

### `params.yaml`

Add:

```yaml
build_tubes:
  min_tube_length: 4
  min_detected_entries: 2
```

The existing `tubes.iou_threshold` / `tubes.max_misses` are reused.

## Testing (minimal for a research spike)

- `tests/test_data.py`:
  - `load_detections` on a 5-col file → `confidence=1.0`
  - `load_detections` on a 6-col file → confidence from column 6
  - `load_detections` on an empty file → `[]`
- `tests/test_tubes.py`:
  - `interpolate_gaps` — interior gap of length 1 → midpoint
  - `interpolate_gaps` — leading and trailing boundary gaps repeat nearest
  - `interpolate_gaps` — every gap entry has `confidence=0.0, is_gap=True`
  - `select_longest_tube` on empty list → `None`
  - `select_longest_tube` tie-break by non-gap count

No end-to-end script test for v1. The notebook is the smoke test.

## Error handling

- Missing `labels/` directory → drop, reason `"no_labels_dir"`
- All-empty label files → drop, reason `"no_detections"`
- Single-frame or too-short sequences → drop, reason `"too_short"`
- Malformed `.txt` lines (wrong column count, non-numeric) → warn, skip line, continue
- Sequence ID collision across WF and FP → fail-loud assertion

## Verification

1. `make lint && make test` pass in `smokeynet-adapted/`.
2. `uv run dvc repro build_tubes@train` and `build_tubes@val` both
   succeed and populate `data/03_primary/tubes/{train,val}/`.
3. `_summary.json` shows a reasonable dropout rate. Pre-spike
   measurement on a 200-sequence FP sample with `iou_threshold=0.2`,
   `max_misses=2`, `min_tube_length=4`: **~94% survival** in both
   train and val (median longest-tube length 10-11 frames). If WF
   dropouts exceed ~10%, loosen `min_tube_length` or
   `min_detected_entries` before freezing the dataset.
4. `notebooks/02-visualize-built-tubes.ipynb` loads a handful of
   `<sequence_id>.json` files and renders correct tube timelines +
   filmstrips for both smoke and FP sequences.

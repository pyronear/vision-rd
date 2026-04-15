> Renamed 2026-04-15: smokeynet-adapted → bbox-tube-temporal. Old paths in this doc reflect the design-time state.

# Render Tubes — Design

**Date:** 2026-04-14
**Status:** Approved, ready for implementation plan
**Scope:** `experiments/temporal-models/bbox-tube-temporal/`

## Goal

Render one PNG per built smoke tube to a label-nested directory tree so
the dataset can be reviewed quickly with a file explorer (or any image
gallery). Reuses the existing `plot_tube_summary` view (timeline +
filmstrip).

## Inputs

```
data/03_primary/tubes/{train,val}/
  <sequence_id>.json    # one tube per surviving sequence
  _summary.json         # ignored
data/01_raw/datasets/{train,val}/{wildfire,fp}/<seq>/images/*.jpg
```

## Outputs

```
data/08_reporting/tubes/{train,val}/
  smoke/<sequence_id>.png
  fp/<sequence_id>.png
```

Each PNG is the `plot_tube_summary` rendering of that sequence's tube:
the timeline strip on top showing detected/gap frames, and the
filmstrip below showing the cropped bbox content per frame.

## Pipeline

At script start, wipe `--output-dir` so stale PNGs from earlier runs
(e.g., for sequences that have since been filtered out) don't linger.
The DVC stage's `outs` already treats the directory as a single
artifact, so this keeps the on-disk state authoritative.

```
For each *.json (excluding _summary.json) in data/03_primary/tubes/<split>/:
  record = load_tube_record(path)
  tube   = tube_from_record(record)
  seq_dir = find_sequence_dir(data/01_raw/datasets/<split>, record["sequence_id"])
  if seq_dir is None: warn + skip
  frame_paths = get_sorted_frames(seq_dir)
  fig = plot_tube_summary(frame_paths, [tube], num_frames=record["num_frames"],
                          tube_labels=[record["label"] == "smoke"],
                          title=f"{record['sequence_id']} [{record['label']}]")
  out_path = data/08_reporting/tubes/<split>/<label>/<sequence_id>.png
  fig.savefig(out_path, dpi=120, bbox_inches="tight")
  plt.close(fig)
```

## Code changes

### `src/bbox_tube_temporal/data.py`

Add:

- `load_tube_record(path: Path) -> dict` — read+parse a tube JSON file.
  Trivial wrapper for `json.loads(path.read_text())`, but gives the
  module a clear API for tube I/O alongside the existing
  `load_detections`.

### `src/bbox_tube_temporal/tubes.py`

Add:

- `tube_from_record(record: dict) -> Tube` — rebuild a `Tube` from the
  dict produced by `_serialize_tube` in `scripts/build_tubes.py`. Pure
  function; no I/O.

### `notebooks/02-visualize-built-tubes.ipynb`

Replace the inline `_record_to_tube` helper with imports of
`load_tube_record` and `tube_from_record`. Behaviour unchanged.

### `scripts/render_tubes.py` (new)

```
uv run python scripts/render_tubes.py \
  --tubes-dir data/03_primary/tubes/<split> \
  --raw-dir   data/01_raw/datasets/<split> \
  --output-dir data/08_reporting/tubes/<split> \
  [--dpi 120]
```

Walks tube JSONs, calls the pipeline above, prints
`[<split>] rendered N/M tubes (smoke=A, fp=B, skipped=K)`.

### `dvc.yaml`

New `render_tubes` stage (after `build_tubes`):

```yaml
render_tubes:
  foreach: [train, val]
  do:
    cmd: >-
      uv run python scripts/render_tubes.py
      --tubes-dir data/03_primary/tubes/${item}
      --raw-dir data/01_raw/datasets/${item}
      --output-dir data/08_reporting/tubes/${item}
    deps:
      - scripts/render_tubes.py
      - src/bbox_tube_temporal/tubes.py
      - src/bbox_tube_temporal/data.py
      - data/03_primary/tubes/${item}
      - data/01_raw/datasets/${item}
    outs:
      - data/08_reporting/tubes/${item}
```

No new params — DPI is fine as a CLI default.

## Testing

Two minimal round-trip tests; no rendered-image assertions.

- `tests/test_data.py`:
  - `load_tube_record` reads a JSON written by `json.dumps` and returns
    an equivalent dict.

- `tests/test_tubes.py`:
  - `tube_from_record` rebuilds a `Tube` whose entries match the input
    record (frame_idx, bbox, is_gap, confidence preserved).

## Error handling

- Source sequence dir not found (record lists a sequence that no longer
  exists under `data/01_raw/datasets/<split>`) → warn, skip, continue.
- Tubes dir empty (no JSONs) → warn, exit 0 (DVC will still track the
  empty output dir).
- Output sub-dirs (`smoke/`, `fp/`) created lazily as the first
  matching tube is encountered.

## Verification

1. `make lint && make test` pass.
2. `uv run dvc repro render_tubes@val` populates
   `data/08_reporting/tubes/val/{smoke,fp}/*.png`.
3. Open `data/08_reporting/tubes/val/smoke/` in a file explorer; PNG
   thumbnails should clearly show smoke trajectories.
4. Open `data/08_reporting/tubes/val/fp/` similarly.
5. `dvc status` reports clean after re-running.

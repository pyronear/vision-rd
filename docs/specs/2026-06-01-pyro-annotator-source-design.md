# Pyro-annotator as a second sequence source — design

**Date:** 2026-06-01
**Experiment:** `experiments/temporal-models/temporal-model-explorer`
**Status:** approved, pending implementation plan

## Goal

Add **pyro-annotator** as a second sequence source in the Temporal Model
Explorer, alongside the existing alert-API import. Pyro-annotator is a
human-curated export shipped as a zip; its value is that every sequence carries
a **real ground-truth label** (smoke / fp / unlabeled, with subtype). The current
alert-API source has mostly `unknown` labels (`is_wildfire` is usually null), so
pyro-annotator gives us a proper **labeled eval set** to see where the temporal
model keeps real smoke and filters false positives.

## Source data: the zip

`data/01_raw/seq_annotation_done_by_label.zip` — **332 sequences**, laid out by
human label:

```
seq_annotation_done_by_label/
  smoke/{wildfire,industrial,other}/seq_<id>/
  fp/{low_cloud,high_cloud,tree,light,lens_droplet,building,antenna,water_body,other}/seq_<id>/
  unlabeled/seq_<id>/
```

Label distribution: fp 273 (low_cloud 213, high_cloud 21, tree 12, light 11,
lens_droplet 7, building 4, antenna 3, water_body 1, other 1), smoke 44
(industrial 31, wildfire 12, other 1), unlabeled 15.

Per sequence:
- `images/detection_<id>.jpg` — full frames, 1280×720.
- `labels/detection_<id>.txt` — ground-truth YOLO bboxes (`class cx cy w h`, normalized).
- `predictor_develop.txt` (all 332) and `predictor_pr366.txt` (250) — baseline
  temporal-predictor runs (per-frame conf + `alert` decision).

**Absent from the zip:** camera id/name, organization, timestamps, any
JSON/CSV/YAML metadata, image EXIF. The `seq_<id>` and `detection_<id>` are real
platform IDs, so this metadata exists on the platform but is only recoverable via
the API.

## Platform API findings

Verified against `https://alertapi.pyronear.org`:

- The **regular** login (`PLATFORM_LOGIN/PASSWORD`) is org-scoped and returns
  **403** for the zip's sequences (they belong to other orgs / older dates).
- The **admin** login (`PLATFORM_ADMIN_LOGIN/PASSWORD`) reads them fine.
- `GET /api/v1/sequences/<id>/detections` returns detections with
  `camera_id`, `created_at`, `bbox`, `url`. `camera_id` is constant per sequence.
- `camera_id` → camera `name` + `organization_id` via `list_cameras`;
  `organization_id` → org `name` via the admin `list_organizations`.

Confirmed end-to-end: e.g. `seq_40972` → camera `nemours-02`, org `sdis-77`, real
timestamps `2026-05-09T15:03 … 15:15`.

**Implication:** the importer can enrich each pyro-annotator sequence with real
`camera_id/name`, `organization_id/name`, and per-frame timestamps, so these
sequences sit in the same organization → camera navigation as alert-API
sequences. Enrichment **requires admin creds**.

## Design

### 1. `source` selector in the app (new)

The sidebar cascade becomes **source → organization → camera → model**.
`results.parquet` already carries a `source` column (`run_models.py` writes
`meta.source`), so this is a thin filter addition — no schema change:

```python
sources = sorted(df["source"].dropna().unique())          # "platform" | "pyro-annotator"
source  = st.sidebar.selectbox("source", sources, key="source") if sources else None
src_df  = df[df["source"] == source] if source else df
orgs    = sorted(src_df["organization_name"].dropna().unique())   # cascades unchanged from here
...
if source:
    view = view[view["source"] == source]                 # added alongside the org/camera filters
```

The word `source` is used everywhere — stored field, parquet column, and UI label
— matching the existing data model (no synonym drift, no data migration). The
existing alert-API source keeps its stored value `"platform"`; the new source is
`"pyro-annotator"`.

### 2. New importer (enrich via admin API)

`src/temporal_model_explorer/import_pyro_annotator.py` (logic) +
`scripts/import_pyro_annotator.py` (thin CLI), mirroring the existing
`import_platform` pair. Per sequence in the extracted zip:

- **Label** from the folder path: `smoke/*` → `smoke`, `fp/*` → `fp`,
  `unlabeled` → `unknown`. `label_detail` = subfolder (`wildfire`, `low_cloud`, …)
  or `None` for `unlabeled`. `label_source = "pyro_annotator_folder"`.
- **Images** copied from the extracted zip into the store `images/` dir — no
  re-download, the exact frames are already present.
- **Enrich** via admin API: `GET /sequences/<id>/detections` (generous limit) →
  build `{detection_id: created_at}` and read `camera_id` from the first
  detection. Map `camera_id` → camera name + org via `list_cameras` /
  `list_organizations`. Order frames by `created_at` (fallback: detection id).
- **meta.json**: `key="pyro_annotator_<id>"`, `source="pyro-annotator"`,
  `sequence_id=<id>`, real `camera_id/name`, `organization_id/name`,
  `started_at` = earliest `created_at`, per-frame `FrameRef.created_at`.

Reuses existing `store.py` (`FrameRef`, `SequenceMeta`, `write_meta`) and
`platform_api.py` (`get_access_token`, `list_sequence_detections`,
`list_cameras`, `list_organizations`, `build_camera_index`, `build_org_index`).

### 3. On-disk layout

```
data/03_primary/sequences/pyro-annotator/<org>/<camera>/seq_<id>/{images/,meta.json}
```

A dedicated top-level `pyro-annotator/` dir avoids collision with the existing
`sis-67/`, `sdis-77/` alert-API trees (the same global seq id would otherwise
clash on disk). `run_models` already rglobs the whole store, so the new
sequences are scored with **zero pipeline change**. DVC-track the new tree with
its own `dvc add data/03_primary/sequences/pyro-annotator`.

### 4. CLI / input handling

- `--src` — path to the extracted zip root (user unzips into `data/01_raw/` first).
- `--out` — store dir, default `data/03_primary/sequences`.
- `--params` — default `params.yaml` (unused for labels; kept for symmetry).
- Reads creds from env: `PLATFORM_API_ENDPOINT`, `PLATFORM_ADMIN_LOGIN`,
  `PLATFORM_ADMIN_PASSWORD`. Errors clearly if admin creds are missing.
- Skips `__MACOSX/` and `.DS_Store` entries.

## Out of scope (per the ground-truth-eval-set goal)

The zip's per-frame bbox `.txt` files and the `predictor_develop` / `predictor_pr366`
baseline outputs are **not** carried into the store. The temporal model runs its
own YOLO and these sequences are used purely as a labeled eval set. The files
stay in `data/01_raw/` for a possible future overlay/comparison feature.

## Error handling

- **Missing admin creds** → fail fast with a clear message (these sequences are
  unreadable without them).
- **A sequence can't be enriched** (deleted / 403 / no detections) → import it
  anyway under `organization_name="unknown"`, `camera_name="unknown"`,
  timestamps `None`, and log a warning. Better to keep the labeled sequence than
  drop it.
- **A zip frame id not present in the API response** → keep the frame,
  `created_at=None` for it.
- **Per-sequence failures** (bad image, network blip) → log and continue, like
  the alert-API importer.

## Testing

- `normalize` of folder path → (label, label_detail) for smoke/fp/unlabeled cases.
- Frame ordering by enriched `created_at` with a `None` fallback.
- meta.json round-trips through `read_meta` with `source="pyro-annotator"`.
- Importer against a tiny fixture tree + a stubbed API client (inject the
  `list_sequence_detections` / camera-index functions, no network), asserting the
  written store layout and meta contents.
- App-side: a unit-level check that the `source` filter narrows the dataframe;
  the cascade reuses existing tested selectbox logic.

## Files touched

- **new** `src/temporal_model_explorer/import_pyro_annotator.py`
- **new** `scripts/import_pyro_annotator.py`
- **edit** `src/temporal_model_explorer/app.py` — add the `source` selectbox + filter
- **new** `tests/test_import_pyro_annotator.py`
- **edit** `README.md` — document the new source + import command
- (data) `dvc add data/03_primary/sequences/pyro-annotator`

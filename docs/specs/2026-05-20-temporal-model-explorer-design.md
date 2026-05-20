# Temporal Model Explorer — Design

- **Date:** 2026-05-20
- **Status:** Approved (design); pending implementation plan
- **Scope:** new experiment `experiments/temporal-models/temporal-model-explorer/`; depends on `lib/bbox-tube-temporal/` (the `bbox-tube-temporal-core` package).

## Goal

A **local** tool to see which camera-event sequences the temporal model would
**KEEP** (raise an alert) vs **DISCARD** (filter as a false positive), and to
**compare models** on the same sequences. Pull sequences from the Pyronear
platform API (and/or load a local annotated dataset), store them locally, run one
or more temporal models, and browse the keep/discard results in a Streamlit app
with filters. R&D tool; not a production service.

## Non-goals

- No production API / deployment / cloud (that direction was dropped).
- No retraining; we only *run* already-packaged models.
- No write-back to the platform; read-only consumption.
- Organization **names** are an optional enrichment (need admin creds or a static
  `org_id → name` map); the tool works fully without them. Admin access is
  obtainable if/when we want org-name grouping.

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | 3-stage pipeline (`import → run → view`) + Streamlit viewer | Matches repo's Kedro-style layers; keeps heavy model work out of the UI; reproducible. |
| Frontend | Streamlit first; **results kept frontend-agnostic** so a tailored app (React/Next) can replace the viewer later | Fast to build now; the import/run layers and result artifacts don't change if we swap UIs. |
| Data sources | Platform API import **and** local annotated zip | Zip gives clean ground truth now; API gives fresh sequences. |
| Platform client | Mirror-copied lean `requests` client (login + sequences + detections) | Verified working with regular creds; no heavy deps, no admin. |
| Comparison framing | Per-sequence **KEEP/DISCARD** + outcome vs ground-truth label, with filters | The actual question: which alerts get filtered, and which errors. |
| Models | Packaged temporal models via the lib (`BboxTubeTemporalModel.from_package`), config-driven & pluggable | Reuses Phase-1 lib; supports >1 model for comparison. |
| Auth/creds | Read from environment (`.envrc`, gitignored): `PLATFORM_API_ENDPOINT`, `PLATFORM_LOGIN`, `PLATFORM_PASSWORD` | Matches existing convention; admin creds not required. |

## Architecture

```
                       ┌──────────────────────────────┐
 platform API ──┐      │  common local sequence store │      ┌──────────┐     ┌───────────┐
                ├─►importers─►  data/03_primary/        ─►│ model    │─►│ results   │─► Streamlit
 local zip ─────┘      │   sequences/<key>/{images,meta}│  │ runner   │  │ (parquet) │     viewer
                       └──────────────────────────────┘      └──────────┘     └───────────┘
```

Components are isolated and independently runnable:

1. **`platform_api.py`** — lean client (mirror-copied): `get_access_token`,
   `list_sequences_for_date`, `list_sequence_detections`, `list_cameras`, and
   (optional, admin) `list_organizations`. Auth via `POST /api/v1/login/creds`;
   bearer header. Camera + `organization_id` come from `/cameras/` (no admin);
   `list_organizations` is only called when admin creds are present.
2. **Importers → common store** (both write `data/03_primary/sequences/<key>/`):
   - **`import_platform.py`** — for a date range (+ optional `--camera-id` filter),
     list sequences, download each detection's full-frame image (`detection.url`),
     write `images/detection_<id>.jpg` + `meta.json`. `key = platform_<sequence_id>`.
   - **`import_local_zip.py`** — extract/adopt `seq_annotation_done_by_label.zip`;
     map folder (`smoke/*`, `fp/*`, `unlabeled`) → label; copy `images/` + write
     `meta.json`. `key = zip_<seq_id>`.
3. **`run_models.py`** — for each sequence in the store, build the ordered
   `list[Frame]`, run each configured model
   (`BboxTubeTemporalModel.from_package(model.zip, device="cpu").predict(frames)`),
   and write one results row per (sequence, model). Skips models whose `model.zip`
   is absent (logs a warning).
4. **`app.py`** — Streamlit viewer over the store + results.

### Common sequence store — `meta.json`

`data/03_primary/sequences/<key>/meta.json`:

```jsonc
{
  "key": "platform_43392",
  "sequence_id": "43392",
  "source": "platform",                // "platform" | "local_zip"
  "camera_id": 12,                     // null if unknown
  "camera_name": "marguareis-01",      // from /cameras/ (no admin); null if unresolved
  "organization_id": 3,                // from /cameras/ (no admin); null if unknown
  "organization_name": "SDIS-07",      // optional: admin /organizations OR params.yaml map; null if unavailable
  "started_at": "2026-05-19T14:10:01Z",// null for zip
  "label": "fp",                       // normalized GT: "smoke" | "fp" | "unknown"
  "label_detail": "other_smoke",       // raw platform is_wildfire OR zip subfolder
  "label_source": "platform_is_wildfire", // or "zip_folder"
  "frames": [                          // ordered oldest→newest
    {"file": "images/detection_2094182.jpg", "detection_id": 2094182, "created_at": "2026-05-19T14:10:01Z"}
  ]
}
```

- **Ordering**: platform → by detection `created_at`; zip → by filename.
- **`label` (tri-state)** drives keep/discard correctness: `smoke` → should KEEP,
  `fp` → should DISCARD, `unknown` → no GT.
- **Platform `is_wildfire` → `label`** is **best-effort & configurable**: values
  containing `smoke` or equal to `wildfire` → `smoke`; other known FP-ish values →
  `fp`; unrecognized → `unknown`. Raw value always kept in `label_detail`. The
  local zip is the gold-standard GT.

**Field provenance by source.** The rich metadata comes from the **API**:
- **platform** → all fields from the API: `sequence_id, camera_id, started_at,
  is_wildfire`(→`label_detail`) from the sequence; `camera_name, organization_id`
  from `/cameras/`; `frames` from the sequence's detections; `organization_name`
  only with admin (or the static map).
- **local_zip** → only `label`/`label_detail` (folder), `frames` (`images/`,
  ordered by filename), and `sequence_id` (`seq_<id>` dir). `camera_id`,
  `camera_name`, `organization_*`, `started_at` are **not in the zip** → `null` by
  default, but can be **optionally enriched** by looking the `sequence_id` up
  against the API (`/sequences/{id}` + `/cameras/`), since the zip's `seq_<id>`
  are platform sequence IDs.

### Model runner — results format

`data/07_model_output/results.parquet` (+ per-sequence `…/<model>/<key>.json` with
full `details`). One row per (sequence, model):

| column | meaning |
|---|---|
| `key, source, sequence_id, camera_id, camera_name, organization_id, organization_name, label, label_detail, n_frames` | from `meta.json` |
| `model` | model name (config key) |
| `decision` | `keep` (is_smoke=True) / `discard` |
| `trigger_frame_index`, `trigger_frame_file` | 0-based index + resolved frame filename (null if discard) |
| `probability` | calibrated prob if the package has a calibrator, else null |
| `outcome` | vs `label`: `kept-smoke` / `discarded-fp` / `discarded-smoke` (🔴) / `kept-fp` (🟠) / `n/a` (unknown label) |
| `runtime_ms` | per-sequence inference time |

Models are configured in `params.yaml` (name → `model.zip` path), defaulting to the
packaged bbox-tube variants under
`experiments/temporal-models/bbox-tube-temporal/data/06_models/<variant>/model.zip`.

### Streamlit app

- **Main table** — one row per sequence; per-model KEEP/DISCARD + trigger + the
  `outcome` chip. Sortable.
- **Sidebar filters** — model decision (KEEP/DISCARD); GT label (smoke/fp/unknown);
  outcome (incl. "errors only", "smoke wrongly discarded", "fp correctly
  discarded", "fp wrongly kept"); station/camera, organization; source; and
  (with >1 model) agreement (agree/disagree).
- **Drill-down** — select a sequence: ordered frame strip with bbox overlay and the
  trigger frame highlighted; per-model panel (KEEP/DISCARD, trigger index,
  probability, raw `details`).

**Frontend-agnostic results.** Streamlit reads only the result artifacts
(`results.parquet`, per-sequence `details` JSON, and the image store) — it never
re-runs models or fetches. All non-UI logic (filtering, `outcome` computation)
lives in `outcomes.py`/`store.py`, not in `app.py`. A future tailored app
(React/Next) can read the same artifacts — via a static JSON export of
`results.parquet` or a thin read-only API added later — without changing the
import/run layers.

## Configuration

Env (from `.envrc`, gitignored): `PLATFORM_API_ENDPOINT`
(`https://alertapi.pyronear.org`), `PLATFORM_LOGIN`, `PLATFORM_PASSWORD`.
Optional `PLATFORM_ADMIN_LOGIN`/`PLATFORM_ADMIN_PASSWORD` enable org-name
enrichment via `/organizations/` (skipped if unset/invalid).
`params.yaml`: model registry (name → package path), platform→label mapping,
optional static `org_id → name` map (fallback when admin is unavailable),
default date range / detections limit.

## Error handling / edge cases

- Missing/invalid creds → import fails fast with a clear message.
- A detection with no `url` or a failed image download → skip that frame, log it,
  continue (a sequence with too few frames still runs; the model yields DISCARD).
- A model's `model.zip` absent (DVC not synced) → skip that model with a warning;
  other models still run.
- Sequence with `< 4` usable frames → model naturally returns DISCARD (no tube).
- Re-running an importer is idempotent per `key` (overwrite/skip existing).

## Testing

- **platform_api**: unit tests with mocked `requests` (login, list sequences,
  list detections) — no live calls in CI.
- **importers**: platform importer against mocked API + a tiny fake image server /
  monkeypatched downloader; zip importer against a tiny synthetic zip fixture →
  assert `meta.json` + `images/` layout and label mapping.
- **runner**: with a **fake `TemporalModel`** (returns known output) over a
  synthetic store → assert results rows + `outcome` logic. A real-model test is
  opt-in (`@pytest.mark.integration`, skipped when `model.zip` absent).
- **app**: pure helper functions (filtering, outcome computation) unit-tested;
  Streamlit UI itself not unit-tested.
- Reuse repo CI patterns (ruff check, ruff format --check, pytest).

## Layout

```
experiments/temporal-models/temporal-model-explorer/
  pyproject.toml        # deps: bbox-tube-temporal-core (path), streamlit, requests, pandas, pyarrow, pyyaml, pillow; dev: pytest, ruff
  Makefile              # install / lint / format / test / app (streamlit run)
  params.yaml           # model registry, label mapping, defaults
  dvc.yaml              # stages: import_local_zip / import_platform / run_models
  src/temporal_model_explorer/
    platform_api.py     # lean platform client (mirror copy)
    store.py            # meta.json read/write + sequence store helpers
    import_platform.py  # platform → store
    import_local_zip.py # annotated zip → store
    run_models.py       # store → results
    outcomes.py         # decision/outcome logic (pure, tested)
    app.py              # Streamlit viewer
  scripts/              # thin CLI wrappers (argparse) calling the modules
  tests/
  data/                 # 01_raw, 03_primary/sequences, 07_model_output (gitignored / DVC)
```

### Pipeline (DVC)

Stages are wired in `dvc.yaml` (matching the other experiments), driven by
`params.yaml`:

- **`import_local_zip`** — dep: the annotated zip → out:
  `data/03_primary/sequences/zip_*`. Deterministic.
- **`import_platform`** — params: date range, cameras → out:
  `data/03_primary/sequences/platform_*`. Param-driven, but it hits the **live
  API**, so a re-run fetches whatever the API currently returns (not
  bit-reproducible); treat the fetched store as a cached snapshot.
- **`run_models`** — deps: the sequence store + the configured `model.zip`(s);
  params: model registry → out: `data/07_model_output/results.parquet` (+
  per-sequence `details` JSON). **Deterministic** given store + packages — the
  cleanest DVC stage, and the one you specifically want tracked.

The Streamlit app is **not** a DVC stage — it's a viewer over `run_models`
outputs. Each module also has a thin `scripts/` CLI wrapper, so every stage runs
standalone (`uv run python scripts/run_models.py …`) or via `uv run dvc repro`.

## Prerequisites

- The lib `lib/bbox-tube-temporal/` (already built).
- At least one packaged `model.zip` available locally (DVC-synced) to run a real
  model; absent models are skipped.
- The annotated zip at
  `experiments/temporal-models/data/seq_annotation_done_by_label.zip` for the local
  source.

## Future work

- Add more `TemporalModel` implementations (other architectures) to the registry.
- Per-station / per-FP-category metrics summaries.
- Optional: cache platform downloads to avoid re-fetching.

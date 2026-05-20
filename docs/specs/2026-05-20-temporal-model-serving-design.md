# Temporal Model Serving — Production API Design

- **Date:** 2026-05-20
- **Status:** Approved (design); pending implementation plan
- **Scope:** `lib/bbox-tube-temporal/` (new), `services/temporal-model-api/` (new). The `bbox-tube-temporal` experiment is **left untouched** — the core is *copied* into the lib, not moved.

## Goal

Run the `bbox-tube-temporal` `vit_dinov2_finetune` model in production as an HTTP
API. The service is the **final check before an alert is raised** on the Pyronear
platform: given the recent frames of a candidate sequence, it returns a smoke /
no-smoke verdict used to filter false positives. It does **not** need to be real
time — a few seconds per request is acceptable.

## Non-goals

- Real-time / streaming inference.
- Reusing the edge device's YOLO bboxes (we re-run the bundled YOLO for parity).
- GPU serving, request batching, autoscaling beyond a single instance.
- A queryable prediction database (S3 JSON only).
- Splitting the service into its own repository (deferred — see *Future work*).

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Frame delivery | API fetches from S3 (`s3://`) or HTTP(S) URLs | Frames already live in object storage the API can read. |
| Peak load | ~1000 req/day (a few/min peak) | One always-on CPU instance, synchronous, no queue. |
| YOLO | Re-run the bundled YOLO from `model.zip` | Train/inference parity; ~2s CPU cost is acceptable. |
| Response model | Synchronous request → verdict | Platform gates the alert on the result; latency budget allows it. |
| Cloud / compute | AWS App Runner (public HTTPS) | Caller is external to AWS; managed TLS + simplest tofu. |
| IaC | OpenTofu | Per user preference. |
| Code placement | Core **copied** into `lib/` (experiment untouched), service → `services/`, monorepo | Avoid touching the working training/DVC pipeline now; accept temporary duplication, dedupe at the later split. |
| Framework | LitServe | Lightning-native ergonomics; team familiarity. |
| Repo strategy | Monorepo now, split later | Defer publish/versioning overhead until ops separation earns it. |
| Genericity | Generic-by-construction against `pyrocore.TemporalModel`; no separate serving lib yet | The generic interface already exists in `pyrocore`; one model in prod → YAGNI on a serving framework. |

## Architecture

### Layering and genericity

The generic temporal-model abstraction already exists and is **not** re-invented
here: `pyrocore` provides the `TemporalModel` ABC (`predict(frames) ->
TemporalModelOutput`, an overridable `load_sequence`, and a `predict_sequence`
template method), with `Frame` and an opaque `TemporalModelOutput.details` dict.

| Layer | Generic? | Home |
|---|---|---|
| Model interface (`TemporalModel`, `Frame`, `TemporalModelOutput`) | Generic | `pyrocore` (exists) |
| bbox-tube inference core (tubes, YOLO companion, ViT patches) | Specific (one impl) | `lib/bbox-tube-temporal/` |
| Serving (HTTP/LitServe, auth, S3 fetch, prediction storage, schemas) | Generic by construction | `services/temporal-model-api/` |

The serving layer is model-agnostic by construction: the request is just frames
(→ `list[Frame]`), the response is a boolean verdict / `trigger_frame_index` /
opaque `details`. The **only** model-specific code in the service is a single loader
seam — `load_model() -> (TemporalModel, model_version)` — invoked in `setup()`.
Everything else is written against the `pyrocore.TemporalModel` ABC, never
naming `BboxTubeTemporalModel` outside that adapter. This keeps a future
extraction into a generic `lib/temporal-serving/` package cheap (see *Future
work*) without paying for it now.

### Repo layout

```
lib/bbox-tube-temporal/                 # NEW — COPY of the 9 inference-core modules
  src/bbox_tube_temporal/               # mirror: same import name + layout (diff-able vs experiment)
    model.py inference.py package.py tubes.py model_input.py
    types.py logistic_calibrator.py details_schema.py temporal_classifier.py
  tests/                                # copies of the tests that exercise these modules
  pyproject.toml                        # distribution name bbox-tube-temporal-core; deps below

experiments/temporal-models/bbox-tube-temporal/   # UNCHANGED — not touched at all
  src/bbox_tube_temporal/ ...           # keeps its own copies; training/eval/DVC code as-is
  # no rename, no import rewrites, no pyproject change

services/temporal-model-api/            # NEW — deployable (depends on the lib only)
  src/temporal_model_api/
    server.py     # LitServer(...).run() entrypoint
    api.py        # LitAPI subclass (setup/decode_request/predict/encode_response)
    schemas.py    # pydantic request/response models
    s3.py         # frame download + prediction-JSON persist
    config.py     # pydantic-settings env config
  tests/
  Dockerfile
  deploy/tofu/    # ECR, App Runner, IAM, Secrets Manager, S3, CloudWatch
  pyproject.toml
```

### Lib dependency set (lean prod runtime)

`pyrocore, torch, torchvision, timm, ultralytics, numpy<2, pillow, pyyaml,
pydantic`. Confirmed by reading the core modules: `model_input` is pure
PIL/numpy (no cv2), `logistic_calibrator.predict_proba` is pure numpy (no
sklearn), ultralytics is imported lazily inside `package._load_yolo`, and no
core module imports lightning. (opencv arrives transitively via ultralytics.)

### Extraction approach — copy, experiment untouched

The 9 core modules use only relative internal imports (verified), so they copy
into the lib with **zero source edits**. We deliberately **copy** rather than move:

- The experiment is **not modified** — no package rename, no import rewrites, no
  `pyproject` change. Its training/DVC pipeline keeps working exactly as-is. This
  is the explicit goal of this approach (lower risk, faster).
- The lib is a **mirror copy**: same import name (`bbox_tube_temporal`) and file
  layout, so `diff` between the two trees detects drift directly. The lib's
  *distribution* name is `bbox-tube-temporal-core` so the two never clash if ever
  co-installed (in practice they aren't — each uv project installs in isolation).
- The lib also ships **copies of the tests** that exercise these modules (parity,
  packaging, inference, tubes, model_input, calibrator, details schema), so the
  lib's copy is independently verified rather than trusted blindly.

**Tradeoff — duplication.** The inference core now lives in two places and can
drift; in the worst case the experiment's inference code changes and the lib's
copy silently loses train/inference parity. Mitigations: the diff-able mirror
layout, the copied parity tests in the lib, and the fact that what actually
changes between model versions is the *weights* (`model.zip`, sha-versioned), not
this loading/inference code, which is stable. Deduplication — the experiment
switching to depend on the lib — is **deferred** to the later split (see *Future
work*) and tracked as known debt.

## API contract

### `POST /predict`

Request:
```jsonc
{
  "sequence_id": "seq-abc123",   // caller id, echoed into storage for audit/correlation
  "camera_id": "marguareis-01",  // optional; used for storage partitioning
  "frames": [                     // ORDERED oldest → newest; client-ordered, API does NOT sort
    {"uri": "s3://pyro-frames/.../f0.jpg"},
    {"uri": "https://.../f1.jpg"}
    // typically ~6-20 recent frames; object form leaves room for optional
    // per-frame metadata later (e.g. edge bbox: {"uri", "bbox"}) without a break
  ]
}
```
- `uri` accepts `s3://` (fetched via the instance IAM role) or `http(s)://`
  (incl. presigned). Mixed schemes allowed. (`file://` is also accepted for
  local integration testing only — never used in production.)
- **List position is the time axis**: the model treats index `i` as frame `i`, so
  the client must order frames oldest → newest. The API does **not** sort — frame
  timestamps are unreliable in pyro-dataset, and TTD is defined positionally
  (`trigger_frame_index × 30s`). No `timestamp` or `frame_id` field: the model
  uses neither, and "which frame triggered" is recovered positionally.

Response:
```jsonc
{
  "is_smoke": true,                    // the verdict (maps from TemporalModelOutput.is_positive)
  "trigger_frame_index": 2,            // 0-based index into the EXECUTED sequence; null if negative
  "trigger_frame_uri": "s3://.../f2.jpg", // originating input frame; null if negative
  "frames_executed": [                 // input frames acted on, in order (post-truncation)
    "s3://pyro-frames/.../f0.jpg", "https://.../f1.jpg", "s3://.../f2.jpg"
  ],
  "model_version": "vit_dinov2_finetune", // friendly variant name
  "model_revision": "a1b2c3d4e5f6",    // model.zip sha256[:12]; audit/repro (see note)
  "details": { /* full BboxTubeDetails: preprocessing (truncation/padding), tubes, logits, decision */ },
  "request_id": "uuid"                 // server-generated
}
```
- **Field naming:** `is_smoke` is the consumer-facing name for the verdict (the
  platform reads it to keep or filter the alert); internally it is
  `TemporalModelOutput.is_positive`. Latency is **not** in the response — it is
  recorded in storage and structured logs instead (server-side observability;
  the caller can time its own HTTP call).
- **Frame order executed on:** `frames_executed` echoes the ordered input URIs
  the model actually ran on (input truncated to `max_frames=20`), so the caller
  sees exactly what was processed.
- **Trigger mapping:** `trigger_frame_index` is relative to the *executed* sequence
  (after truncation and inference-time padding); `details.preprocessing`
  (`num_truncated`, `padded_frame_indices`) disambiguates. The API resolves it back
  to the originating real input frame as `trigger_frame_uri` so the caller never
  has to reason about padding.
- **Version vs. sha:** the sha is **not** baked into `model_version`. `model_version`
  is the human-friendly variant; `model_revision` is the `model.zip` sha256[:12],
  which traces a prediction to the exact artifact across retrains/repackaging. It
  is primarily an *audit* field — always recorded in storage; keep it in the
  response for full traceability, or drop it from the response if the caller only
  needs the friendly name.

### `GET /health`

Readiness + liveness for the App Runner health check. Returns `503` until the
model is loaded in `setup()`, `200` after.

### Auth

Mandatory `X-API-Key` header validated against a value stored in AWS Secrets
Manager and injected as an env var. Enforced via LitServe's underlying FastAPI
app (middleware/dependency on `LitServer.app`). App Runner provides TLS.

> **To confirm at impl time:** the exact LitServe hook for request auth
> (custom middleware on the FastAPI app vs. a built-in mechanism).

## LitServe `LitAPI` mapping

- `setup(self, device)`: call the **loader seam** `load_model() -> (TemporalModel,
  model_version)` — the only bbox-tube-specific code in the service, which wraps
  `BboxTubeTemporalModel.from_package("/opt/model/model.zip", device="cpu")` and
  `model_revision = sha256(model.zip)[:12]`. Then set torch threads to a sane
  default (detected vCPUs); init boto3 clients; run one synthetic warm-up
  `predict`. The stored model is typed as `pyrocore.TemporalModel`.
- `decode_request(self, request)`: validate against the request schema;
  parallel-download frames (ThreadPoolExecutor) **in request order (no sorting)**
  to a per-request `TemporaryDirectory`; build `list[Frame]` with
  `frame_id` = URI basename and `timestamp=None` (the model uses neither).
  Returns the frames + a temp dir handle + echo metadata.
- `predict(self, x)`: `model.predict(frames)` → `TemporalModelOutput`.
- `encode_response(self, output)`: build the response JSON; **persist the
  prediction to S3 inline** (`PutObject`, ~tens of ms); clean up the temp dir in
  a `finally`.

Single LitServe worker (model loaded once; CPU-bound work serializes on the GIL,
which is fine at this volume). No batching.

## Inference flow (per request)

1. Auth check (`X-API-Key`).
2. Validate body (pydantic) → `422` on malformed input.
3. Parallel-download frames **in the given order (no sort)** to a temp dir →
   `422` naming the failed frame on a download/missing-object error.
4. `model.predict(frames)` on the startup-loaded model.
5. Build the response: resolve `trigger_frame_index` → `trigger_frame_uri` via the
   padding bookkeeping in `details.preprocessing`; echo `frames_executed`.
6. Persist prediction JSON to S3; emit a structured log line.
7. Return the response; temp dir auto-cleaned.

**Latency budget:** ~3s YOLO+ViT + ~0.5-1.5s parallel S3 download ≈ p50 ~3-4s,
p95 ~7-8s. Keep the predictions bucket and (ideally) the frames bucket in the
App Runner region. App Runner request timeout set comfortably above p95.

## Prediction storage (S3)

One object per request:

```
s3://<PREDICTIONS_BUCKET>/predictions/dt=YYYY-MM-DD/camera=<camera_id>/<HH-MM-SS.mmmZ>_<request_id>.json
```

- **Partition order — `dt` first (recommended):** standard for append-only event
  logs. It lets a date-based S3 lifecycle/retention rule expire old predictions
  with one prefix, and spreads writes across daily prefixes (no per-camera hot
  prefix). Because these are Hive-style partitions, Athena/Glue can still filter
  by `camera` regardless of order — so flip to `camera=…/dt=…` only if your
  dominant access pattern is browsing one camera in the S3 console.
- **Filename** carries a sortable intra-day UTC time prefix (`HH-MM-SS.mmmZ`)
  before `request_id`, so objects sort chronologically within a partition and the
  inference time is visible without opening the file.

Payload: echoed request (`sequence_id`, `camera_id`, ordered frame URIs),
full output (`is_smoke`, `trigger_frame_index`, `trigger_frame_uri`,
`frames_executed`, `details`), `model_version` + `model_revision` (sha),
latency breakdown (recorded here even though it is not in the response), and the
full-precision UTC inference timestamp. Date + camera
partitioning makes it directly Athena-queryable and doubles as an audit trail and
future active-learning corpus. No database.

## Packaging

- Multi-stage `Dockerfile`. Build context = **repo root** (so the path-dep lib is
  in context). `uv sync` the service including the lib. **CPU-only torch wheel**
  (via the CPU extra-index) to keep the image ~1.5 GB.
- `COPY` the synced `vit_dinov2_finetune/model.zip`
  (`experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip`)
  into the image at the fixed path `/opt/model/model.zip`. The image is the
  immutable, versioned artifact; `model_revision` is derived from the baked zip's
  sha256 at startup.
- The build assumes `model.zip` is present locally (DVC synced manually — the
  build never runs `dvc pull`).

> Alternative considered: pull `model.zip` from S3 at startup (smaller image,
> hot-swap). Rejected for now in favor of immutable images; revisit if model
> updates become frequent.

## Deployment (OpenTofu, App Runner)

`deploy/tofu/` provisions:
- `aws_ecr_repository` — the service image.
- `aws_apprunner_service` — image from ECR, **2 vCPU / 4 GB**, port 8000, health
  check path `/health`, env config, instance role. Auto-deploy on ECR push
  (optional). **Sizing:** ViT-S/14 + YOLO weights are small; the load is torch's
  CPU runtime (~1–1.5 GB) plus per-request full-frame arrays + patch tensors
  (tens of MB). 4 GB should hold with headroom, but peak RSS under real frames is
  the one thing to **validate in the phase-3 smoke test**. If it's tight, App
  Runner allows **2 vCPU / 6 GB** without changing vCPU — a clean fallback.
  (App Runner is CPU-only, so the CPU-only torch wheel is mandatory regardless.)
- `aws_iam_role` (App Runner **instance** role): read frames bucket, write
  predictions bucket, `secretsmanager:GetSecretValue` on the API-key secret.
- `aws_iam_role` (App Runner **access** role): pull from ECR.
- `aws_secretsmanager_secret` (+ version) for the `X-API-Key`.
- `aws_s3_bucket` for predictions (+ optional lifecycle policy).
- `aws_cloudwatch_log_group` (App Runner logs to CloudWatch).

**Cost:** ~$20-30/mo (4 GB memory billed 24/7; 2 vCPU billed only during the
~3s of active compute per request).

## Configuration (env / Secrets Manager)

| Var | Required | Purpose |
|---|---|---|
| `PREDICTIONS_BUCKET` | yes | S3 bucket for prediction JSON. |
| `API_KEY` | yes | From Secrets Manager; checked against `X-API-Key`. |
| `FRAMES_BUCKET` | no | Restrict `s3://` reads; otherwise IAM-scoped. |
| `LOG_LEVEL` | no | Structured logging level (default `INFO`). |

Deliberately **not** env vars:
- **Model path** — the `model.zip` is baked into the image at a fixed constant
  path (`/opt/model/model.zip`); no need to configure it.
- **Torch threads** — set in code to a sane default (detected vCPUs); revisit only
  if container thread oversubscription shows up.
- **AWS region** — resolved via the standard AWS SDK chain; tofu sets the standard
  `AWS_REGION` on the App Runner service, so the app needs no bespoke setting.

## Error handling / edge cases

All errors share one envelope so the caller parses failures uniformly:

```jsonc
{
  "error": {
    "code": "frame_fetch_failed",     // stable machine-readable code
    "message": "could not fetch frame 3 (s3://.../f3.jpg): NoSuchKey",
    "request_id": "uuid",             // correlates with logs/storage
    "context": { "frame_index": 3, "uri": "s3://.../f3.jpg" }  // optional, case-specific
  }
}
```

| Case | HTTP | `code` | Notes |
|---|---|---|---|
| Missing/invalid `X-API-Key` | 401 | `unauthorized` | Checked before any work. |
| Malformed body / bad schema | 422 | `invalid_request` | From pydantic; `context` carries field errors. |
| Too many frames (> hard cap, e.g. 60) | 422 | `too_many_frames` | Cheap guard against abuse; normal >20 is silently truncated, not an error. |
| Frame download fails (missing key / bad URL / scheme) | 422 | `frame_fetch_failed` | `context` names the failing `frame_index` + `uri`. |
| Model still loading | 503 | `model_not_ready` | `/health` stays not-ready until `setup()` completes. |
| Unexpected server fault | 500 | `internal_error` | Generic; real cause only in logs, keyed by `request_id`. |
| Request exceeds App Runner request timeout | 504 | (platform) | Set the timeout comfortably above p95 (~8s) so this is rare. |

**Not errors (normal `200`):**
- `< 4` usable frames or empty input → `is_smoke=false` (the model yields no
  tube). No special-casing.
- A valid sequence with no smoke → `is_smoke=false`.

Idempotency is not required at this volume; `request_id` is server-generated and
`sequence_id` (caller-supplied) supports dedup/joins in storage.

## Testing strategy

- **Lib:** copies of the parity / packaging / inference tests ship with the
  copied modules and must stay green (the copy must match `predict()` outputs).
- **Service unit:** schema validation; S3 fetch (moto/mock); prediction-key
  layout; auth; the full error table (each `code` → status); health.
- **Service integration (fake model, runs in CI):** LitServe / FastAPI
  `TestClient` against `/predict` with a **fake `TemporalModel`** injected in
  `setup()` returning a known output — verifies wiring without loading the real
  ViT or needing `model.zip` (DVC/S3-tracked, absent in CI).
- **Service integration (real model + real sequences, opt-in / skipped in CI):**
  a `@pytest.mark.integration` test that loads the real `model.zip` and runs a
  few **sample sequences pulled from the experiment's data** (at least one
  known-smoke and one known-FP), asserting the expected `is_smoke` verdict
  end-to-end. To exercise the HTTP path without S3, the frame fetcher also
  accepts `file://` URIs (local-only convenience); alternatively the sample
  frames are uploaded to a moto S3 bucket. The test **skips** when the
  `model.zip` or sample sequences are absent (i.e. always in CI), and is runnable
  locally by a contributor who has DVC-synced both.
- Reuse the repo's CI patterns (ruff check, ruff format --check, pytest) for both
  the new lib and the new service.

## Phasing (for the implementation plan)

1. **Copy core → `lib/bbox-tube-temporal/`** (9 modules + their tests, mirror
   layout, distribution `bbox-tube-temporal-core`); lib test suite green.
   Experiment is **not touched**.
2. **Build the LitServe service** (`api.py`, `s3.py`, `schemas.py`, `config.py`,
   `server.py`) + tests with the fake model.
3. **Dockerfile + `deploy/tofu/`**; build, push, deploy; smoke-test against a
   real `model.zip`.

## Future work

- **Generic serving lib:** when a *second* temporal model goes to production,
  extract the serving machinery (LitServe app, auth, S3 fetch, prediction
  storage, schemas) into `lib/temporal-serving/` with a per-model loader adapter.
  Cheap because the service already targets the `TemporalModel` ABC behind the
  `load_model()` seam.
- **Dedupe the copied core:** retire the duplication by having the experiment
  depend on `lib/bbox-tube-temporal/` (rename its package + rewrite ~39 import
  sites) instead of keeping its own copy. Deferred from now to avoid touching the
  working training/DVC pipeline.
- **Repo split:** when ops/security separation earns it, publish the lib (and
  `pyrocore`) as versioned wheels and move the service to its own `pyro-*` repo.
- **GPU / batching:** trivial LitServe toggles if volume ever grows.
- **Edge-bbox reuse:** skip server-side YOLO by accepting edge bboxes (requires
  matching the edge YOLO and revalidating numbers).

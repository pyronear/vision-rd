# Logistic calibrator deployment

Status: design
Date: 2026-04-17

## Context

The precision investigation (`data/08_reporting/precision_investigation/summary.md`)
established that a sequence-level logistic calibrator — fitted on
`(logit, log_len, mean_conf, n_tubes)` features of the max-logit tube —
yields the best operating point for `vit_dinov2_finetune`:
P=0.974 / R=0.950 / F1=0.962 on val, surpassing every other inference
strategy. `scripts/analyze_variant.py` already fits this model offline
(commit `a86f668`) and writes the weights to
`data/08_reporting/variant_analysis/<variant>/platt_model.json`.

What is still missing is **deployment**: `BboxTubeTemporalModel.predict()`
still decides with `max_logit >= threshold`; the fitted calibrator weights
never flow into `package.zip`. The automated-variant-analysis spec
(`docs/specs/2026-04-17-automated-variant-analysis.md`) explicitly deferred
this to a follow-up PR. This doc is that follow-up.

A secondary goal is **naming cleanup**. The investigation called the
intervention "Platt re-calibration", but classic Platt scaling is a
univariate transform of a raw score; what we ship is a *multivariate
logistic regression* on four features. All forward-looking code and
active specs adopt the more accurate name `logistic_calibrator` /
`LogisticCalibrator` / `aggregation: "logistic"`.

## Goal

Wire the existing calibration recipe into production inference with zero
methodology changes. Same features, same model class, same thresholding
approach. No re-fit sweeps, no reliability-diagram validation, no new
metrics — those belong to a later PR if needed.

## Non-goals

- Refitting methodology changes (cross-validation, held-out calibration
  sets, reliability diagrams, ECE / Brier) — covered by a potential
  Option B follow-up.
- Updating `gru_convnext_finetune`'s decision rule. The investigation
  showed Platt / logistic calibration hurts GRU recall under C1+pad;
  GRU stays on `aggregation: "max_logit"`.
- Rewriting the precision-investigation summary's prose. That document
  records a completed investigation where the intervention was named
  "Platt re-calibration"; it gets a single-line footnote pointing at
  the productized name, nothing more.

## Design

### Architecture at a glance

**Package-time** (runs once per variant, inside `scripts/package_model.py`):
when `aggregation == "logistic"`, the packager runs the full inference
pipeline (YOLO → tracking → classifier) on raw `train` data using the
checkpoint it is about to ship, fits a `sklearn.LogisticRegression` on
the resulting records, runs the same pipeline on `val` to calibrate a
probability threshold via the existing `calibrate_threshold()`, and
embeds the calibrator weights + calibrated threshold in `package.zip`.

**Runtime** (per sequence, inside `BboxTubeTemporalModel.predict`):
`load_model_package` parses `logistic_calibrator.json` from the zip into
a pure-numpy `LogisticCalibrator`; `pick_winner_and_trigger` branches on
`decision.aggregation`. When `"logistic"`, it extracts four features
from the max-logit tube, computes
`prob = sigmoid(features @ coefs + intercept)`, and fires when
`prob >= decision.logistic_threshold`.

**No sklearn at runtime.** Fitted weights serialize to JSON; inference
is pure numpy. A fit-time parity check and load-time sanity-check pairs
guarantee the numpy implementation matches sklearn exactly for the
weights we ship.

### Two modules, strict runtime / fit separation

Split along the sklearn boundary so the runtime path never imports
sklearn (project convention: module-level imports only, no lazy imports
except the ultralytics exception in `package.py`).

**`src/bbox_tube_temporal/logistic_calibrator.py`** — runtime + serialization.
Imports: numpy, dataclasses, json. **No sklearn.**

```python
@dataclass(frozen=True)
class LogisticCalibrator:
    features: list[str]              # ["logit", "log_len", "mean_conf", "n_tubes"]
    coefficients: np.ndarray         # shape (n_features,)
    intercept: float
    sanity_checks: list[dict]        # [{"features": [...], "prob": 0.42}, ...]

    @classmethod
    def from_json(cls, path: Path) -> "LogisticCalibrator": ...

    def to_json(self, path: Path) -> None: ...

    def predict_proba(self, features_row: np.ndarray) -> float:
        z = float(features_row @ self.coefficients) + self.intercept
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def verify_sanity_checks(self, atol: float = 1e-6) -> None:
        # raises ValueError if any stored (features, prob) pair does not
        # round-trip through predict_proba within atol.
        ...


def extract_features(winner_tube: dict, n_tubes: int) -> np.ndarray:
    # (logit, log_len, mean_conf, n_tubes). Consolidates the tube_len /
    # tube_mean_conf helpers currently inlined in scripts/analyze_variant.py.
    ...
```

**`src/bbox_tube_temporal/logistic_calibrator_fit.py`** — package-time fitter.
Imports: numpy, sklearn.linear_model at module top, plus
`LogisticCalibrator` and `extract_features` from the sibling module.

```python
def fit(records: list[dict]) -> LogisticCalibrator:
    # Fits binary LogisticRegression(max_iter=1000, C=1.0), parity-tests
    # numpy vs sklearn predict_proba on the training rows
    # (np.allclose atol=1e-6), samples 3 records to persist as
    # sanity_checks, returns a LogisticCalibrator.
    ...
```

`scripts/package_model.py` and `scripts/analyze_variant.py` both import
`logistic_calibrator_fit.fit`. `src/bbox_tube_temporal/model.py` and
`src/bbox_tube_temporal/package.py` import only from
`logistic_calibrator`. The runtime never pulls in sklearn.

Record schema matches `predictions.json`'s per-sequence entry:
`{"label": "smoke"|"fp", "kept_tubes": [{"logit": float, "start_frame": ..., "end_frame": ..., "entries": [...]} ...]}`.

### Packaging flow

`scripts/package_model.py` and `src/bbox_tube_temporal/val_predict.py`
become:

1. Load classifier checkpoint + YOLO (unchanged).
2. Read `variant_cfg["aggregation"]` from `params.yaml` (default
   `"max_logit"`).
3. Calibrate `decision.threshold` via the existing single-tube val
   inference + `calibrate_threshold()` (unchanged).
4. **If `aggregation == "logistic"`:**
   a. Run full-pipeline inference on `data/01_raw/datasets/train/` using
      the same YOLO+tracking+classifier stack as `evaluate_packaged`.
      Collect records with `kept_tubes` structure.
   b. `LogisticCalibrator.fit(train_records)` — fits sklearn, parity-tests
      numpy vs sklearn, captures three sanity-check pairs.
   c. Run full-pipeline inference on `data/01_raw/datasets/val/`.
      Compute calibrated probs via `calibrator.predict_proba_batch`.
   d. `calibrate_threshold(calibrated_probs, labels, target_recall)` →
      `logistic_threshold`.
5. Build config:
   ```yaml
   decision:
     aggregation: "logistic"       # or "max_logit"
     threshold: <calibrated>       # used when aggregation == "max_logit"
     logistic_threshold: <calibrated>  # used when aggregation == "logistic"
     target_recall: 0.95
     trigger_rule: "end_of_winner"
   ```
6. Write `logistic_calibrator.json` into `package.zip`; extend
   `manifest.yaml` with a `logistic_calibrator:` pointer.

When `aggregation == "max_logit"` everything above collapses back to
today's behavior — no full-pipeline inference, no calibrator file.

### Runtime flow

`src/bbox_tube_temporal/package.py` (`load_model_package`):

- If manifest has a `logistic_calibrator` entry, extract the JSON and
  construct a `LogisticCalibrator`. Call `verify_sanity_checks()`
  immediately (fails loud if sklearn-version drift or JSON tampering
  breaks parity).
- Attach as `ModelPackage.calibrator` (`None` when absent).

`src/bbox_tube_temporal/inference.py` (`pick_winner_and_trigger`):

```python
winner = max(kept_tubes, key=lambda t: t.logit)
if decision["aggregation"] == "max_logit":
    fires = winner.logit >= decision["threshold"]
elif decision["aggregation"] == "logistic":
    features = extract_features(winner, n_tubes=len(kept_tubes))
    prob = calibrator.predict_proba(features)
    fires = prob >= decision["logistic_threshold"]
else:
    raise ValueError(...)
```

`end_of_winner` trigger-frame logic is unchanged — we only change
*whether* a trigger fires, not *which frame* it fires on.

### Naming rename

Forward-looking code adopts `logistic_calibrator` consistently:

- New module `src/bbox_tube_temporal/logistic_calibrator.py`.
- `scripts/analyze_variant.py`:
  - `fit_platt_model` → `fit_logistic_calibrator` (thin wrapper around
    the shared `LogisticCalibrator.fit`).
  - `evaluate_platt` → `evaluate_calibrator`.
  - Output filename `platt_model.json` → `logistic_calibrator.json`.
  - Report section header: `## 6. Platt re-calibration (fit on train)` →
    `## 6. Logistic calibration (fit on train)`.
- `docs/specs/2026-04-17-automated-variant-analysis.md`: rewrite "Platt"
  references to "logistic calibrator", with a one-line note:
  *"Originally named 'Platt re-calibration'; renamed for accuracy
  (multivariate logistic regression, not univariate Platt scaling)."*
- `dvc.yaml`: update the `analyze_variant` `outs:` entry from
  `platt_model.json` to `logistic_calibrator.json`.

Historical artifacts stay intact:

- `data/08_reporting/precision_investigation/summary.md` and
  `data/08_reporting/precision_investigation/*.md` record the completed
  investigation faithfully. Append a single footnote at the end of
  `summary.md`: *"Productized as `LogisticCalibrator` — see
  `docs/specs/2026-04-17-logistic-calibrator-deployment-design.md`."*
- Already-generated `data/08_reporting/variant_analysis/<variant>/analysis_report.md`
  files are not rewritten; they age out on the next `analyze_variant`
  run.

### `params.yaml` changes

Per-variant aggregation knob (scoped inside the existing
`train_<variant>` block to keep the schema consistent with other
variant-specific flags):

```yaml
train_gru_convnext_finetune:
  ...
  aggregation: "max_logit"

train_vit_dinov2_finetune:
  ...
  aggregation: "logistic"
```

Default if absent: `"max_logit"`.

### DVC stage changes

`package` stage gains:

- `deps`:
  - `data/01_raw/datasets/train/` (only read when `aggregation: "logistic"`;
    DVC always tracks it regardless, harmless)
  - `data/01_raw/datasets/val/`
  - `src/bbox_tube_temporal/logistic_calibrator.py`
  - `src/bbox_tube_temporal/inference.py` (already imported transitively;
    make explicit)
- `params`: add `train_<variant>.aggregation`.
- `outs`: unchanged (`data/06_models/<variant>/model.zip`).

`analyze_variant` stage: `outs` path renamed to
`logistic_calibrator.json`. Still downstream of `evaluate_packaged`; no
new dependency relationships.

No new stages.

## Critical files

| Purpose | File |
|---|---|
| New: pure-numpy calibrator, serialization, feature extraction | `src/bbox_tube_temporal/logistic_calibrator.py` |
| New: sklearn-based fitter (package-time only) | `src/bbox_tube_temporal/logistic_calibrator_fit.py` |
| Extend: package-time fit + threshold calibration | `scripts/package_model.py`, `src/bbox_tube_temporal/val_predict.py` |
| Extend: zip manifest + loader | `src/bbox_tube_temporal/package.py` |
| Extend: decision branch | `src/bbox_tube_temporal/inference.py` |
| Extend: propagate calibrator through `predict()` | `src/bbox_tube_temporal/model.py` |
| Rename: consume shared fitter + runtime module | `scripts/analyze_variant.py` |
| Config | `params.yaml`, `dvc.yaml` |
| Spec update | `docs/specs/2026-04-17-automated-variant-analysis.md` |
| Footnote | `data/08_reporting/precision_investigation/summary.md` |

## Testing

- `tests/test_logistic_calibrator.py` (runtime module):
  - Round-trip `to_json` / `from_json` preserves weights + sanity checks
    exactly.
  - `predict_proba` and `predict_proba_batch` agree on single rows.
  - `extract_features` on a hand-constructed tube yields the expected
    values (including `log_len = log1p(end - start + 1)` and
    `mean_conf = mean([entry["conf"] for entry in entries])` — verify
    the exact entry-level field name against
    `scripts/analyze_variant.py:tube_mean_conf` during implementation).
  - `verify_sanity_checks` passes on correct weights, raises on tampered
    weights.
- `tests/test_logistic_calibrator_fit.py` (fitter):
  - Fit on a synthetic 200-record dataset; assert sklearn and numpy
    `predict_proba` agree to atol 1e-6 on the training rows (the
    parity check we rely on for deployment correctness).
  - The returned `LogisticCalibrator` has exactly three sanity-check
    pairs and `verify_sanity_checks()` passes on them.
- `tests/test_package.py`:
  - Build a package with `aggregation: "logistic"`; assert the zip
    contains `logistic_calibrator.json` and the manifest points to it.
  - Load the package; assert `ModelPackage.calibrator` is non-None and
    passes its sanity checks.
  - Loading with a corrupted calibrator JSON raises.
- `tests/test_inference.py`:
  - `aggregation: "max_logit"` branch is byte-identical to today's
    decision logic (regression guard).
  - `aggregation: "logistic"` branch with a known calibrator and known
    tube inputs produces the expected fire/no-fire decision and trigger
    frame index.

## Verification (end-to-end)

1. `cd experiments/temporal-models/bbox-tube-temporal`.
2. `make lint && make test` — all green.
3. Set `train_vit_dinov2_finetune.aggregation: "logistic"` in
   `params.yaml` (already the target config).
4. `uv run dvc repro package@vit_dinov2_finetune` — confirm the stage
   now emits `data/06_models/vit_dinov2_finetune/model.zip` with a
   `logistic_calibrator.json` inside, and logs a `logistic_threshold`
   at or near 0.50.
5. `uv run dvc repro evaluate_packaged@{variant:vit_dinov2_finetune,split:val}`
   — expect val metrics consistent with the investigation's Platt row:
   **P ≥ 0.97, R ≥ 0.94, F1 ≥ 0.96**. Minor drift is acceptable because
   the fit uses package-time full-pipeline inference rather than
   `analyze_variant`'s post-hoc predictions — same recipe, slightly
   different input rows.
6. `uv run dvc repro package@gru_convnext_finetune` — regression guard:
   `aggregation: "max_logit"`, no calibrator in the zip, val metrics
   unchanged from current baseline.
7. Inspect `data/08_reporting/variant_analysis/vit_dinov2_finetune/`:
   `logistic_calibrator.json` replaces `platt_model.json`; report uses
   "Logistic calibration" header.

## Out of scope (future work)

- Proper calibration methodology (train/val split discipline beyond
  what `calibrate_threshold` already provides, cross-validated fits,
  reliability diagrams, ECE / Brier metrics). Planned as a follow-up
  PR if the simple deployment shows drift over time.
- Re-enabling `logistic` for `gru_convnext_finetune`. Requires a
  GRU-specific re-fit study first; current evidence says `max_logit`
  is better for GRU.
- Generalizing the calibrator to other feature sets (e.g., adding
  detector-score variance, tube-age). Easy to extend `extract_features`
  + `params.yaml` schema once there is evidence for additional
  features.

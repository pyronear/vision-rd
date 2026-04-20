# Migrate TTD from seconds to frame index

Status: design (not yet implemented)
Date: 2026-04-20

## Goal

Replace time-to-detection (TTD) values computed from frame filename
timestamps with TTD in **frames** (i.e., `trigger_frame_index`) across
the leaderboard and all per-experiment evaluators. Headline median TTDs
of 7s on the current leaderboard are artifacts of unreliable timestamps
in the pyro-dataset, not real detection latencies.

## Context

### The bug

All current TTD computations in this repo follow the same shape:

```python
ttd_seconds = (frames[trigger_frame_index].timestamp - frames[0].timestamp).total_seconds()
```

This subtracts two parsed filename timestamps (see
`lib/pyrocore/src/pyrocore/types.py:14` for the timestamp parser used
downstream for sort ordering). Filename timestamps in the pyro-dataset
sequential test set are known to be buggy: observed pyronear-force
sequences show 1–3 second gaps (and intra-sequence outliers up to 600
seconds) where the true production cadence is 30 seconds per frame.

Concrete evidence from the current leaderboard
(`experiments/temporal-models/temporal-model-leaderboard/data/08_reporting/leaderboard.txt`):

```
Rank  Model                                     Mean TTD (s)  Median TTD (s)
1     bbox-tube-temporal-vit-dinov2-finetune    89.1          29.0
3     bbox-tube-temporal-gru-convnext-finetune  55.5          7.0
4     pyro-detector-baseline                    27.0          7.0
```

Per-source breakdown (from per-model `results.json` stratified by
sequence_id prefix) shows the two source datasets produce incomparable
numbers:

```
                     hpwren (60s cadence)  pyronear (1–3s gaps)
pyro-detector         median=179s           median=7s
gru-convnext          median=120s           median=7s
vit-dinov2            median=120s           median=23.5s
fsm-tracking          median=300s           median=53s
mtb-change            median=120s           median=23s
```

The 7s median is driven entirely by pyronear sequences where the first
few frames happen to be ~1–3 seconds apart in the filename timestamps.
It is not a detection-latency measurement.

### The four duplicated implementations

The same buggy pattern is copy-pasted across five places:

- `experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/runner.py:72` (`_compute_ttd`)
- `experiments/temporal-models/bbox-tube-temporal/src/bbox_tube_temporal/protocol_eval.py:62`
- `experiments/temporal-models/pyro-detector-baseline/src/pyro_detector_baseline/evaluator.py:30` (`_extract_ttd_seconds`)
- `experiments/temporal-models/tracking-fsm-baseline/src/tracking_fsm_baseline/evaluator.py:33` (`_extract_ttd_seconds`)
- `experiments/temporal-models/mtb-change-detection/src/mtb_change_detection/evaluator.py:33` (`_extract_ttd_seconds`)

### Verified invariants

`trigger_frame_index` is a 0-based index into the frame list each model
receives (`list_sequences` + `get_sorted_frames`), consistent across all
four model families. Grep of `trigger_frame_index=` assignments in
`experiments/temporal-models/*/src/*/model.py` confirms this. So for a
true positive: **TTD in frames = `trigger_frame_index` directly**. No
computation, no helper function.

## Non-goals

- **No shared TTD helper in pyrocore.** The "computation" collapses to
  an identity function for TPs; extracting a helper hides a trivial
  value behind an import. One docstring line on
  `TemporalModelOutput.trigger_frame_index` codifies the convention
  (see Design / pyrocore below).
- **No per-source stratification** (pyronear vs hpwren) in leaderboard
  reporting. Worth considering separately, but orthogonal to this fix.
- **No percentile columns** (P90, max). Mean and median only, same as
  today.
- **No dataset-side fix.** Filename timestamps in pyro-dataset remain
  buggy; that is an upstream repo concern.
- **No breaking change to `Frame.timestamp`.** It stays present and is
  still used for sorting frame paths. We only forbid *subtracting* two
  such timestamps for durations.

## Design

### Convention

> **TTD in frames = `TemporalModelOutput.trigger_frame_index`** for a
> true positive (ground truth wildfire AND model predicted positive AND
> `trigger_frame_index is not None`). `None` otherwise.
>
> Do not compute TTD from frame filename timestamps. The pyro-dataset
> has known timestamp bugs; timestamps are ordering-only.

This convention is documented in the docstring of
`TemporalModelOutput.trigger_frame_index`
(`lib/pyrocore/src/pyrocore/types.py:34-36`) — one of the few shared
touchpoints all experiments already depend on.

### File-by-file changes

#### `lib/pyrocore/src/pyrocore/types.py`

Add two sentences to the `trigger_frame_index` docstring on
`TemporalModelOutput`:

> Time-to-detection in frames equals this value for a true positive.
> Do not compute TTD by subtracting frame filename timestamps — they
> are unreliable in the pyro-dataset.

No code changes.

#### `experiments/temporal-models/temporal-model-leaderboard/`

- **`src/.../types.py`**
  - `SequenceResult.ttd_seconds: float | None` → `ttd_frames: int | None`
  - `ModelMetrics.mean_ttd_seconds: float | None` → `mean_ttd_frames: float | None`
  - `ModelMetrics.median_ttd_seconds: float | None` → `median_ttd_frames: float | None`
- **`src/.../runner.py`**
  - Delete `_compute_ttd` (lines 72–97).
  - Inline in `evaluate_model`:
    ```python
    ttd_frames = (
        output.trigger_frame_index
        if ground_truth and output.is_positive and output.trigger_frame_index is not None
        else None
    )
    ```
  - `frames` is still needed (model requires it for predict) but no
    longer read for TTD.
- **`src/.../metrics.py`**
  - Comprehension reads `r.ttd_frames` (the identity-mapped field) and
    aggregates via `statistics.mean` / `statistics.median`. Mean/median
    stay as floats (with `statistics.median` of ints returning the
    midpoint as-is for odd-length lists or average of two middles for
    even-length lists).
  - Output field names: `mean_ttd_frames`, `median_ttd_frames`.
- **`src/.../leaderboard.py`**
  - `_LOWER_IS_BETTER`: replace `"mean_ttd_seconds"` / `"median_ttd_seconds"` with `"mean_ttd_frames"` / `"median_ttd_frames"`.
  - Column headers: `"Mean TTD (frames)"` / `"Median TTD (frames)"`.
  - Formatting: keep `.1f` (median is float in general).
- **`tests/test_runner.py`**, **`tests/test_metrics.py`**,
  **`tests/test_leaderboard.py`**
  - Update assertions from seconds to frames. Existing fixtures use 30s
    spacing so `ttd_seconds == 120.0` becomes `ttd_frames == 4` for a
    trigger at index 4.

#### `experiments/temporal-models/{pyro-detector-baseline, tracking-fsm-baseline, mtb-change-detection}/`

For each of these three:

- **`src/.../evaluator.py`**
  - Delete `_extract_ttd_seconds` and its timestamp-subtraction loop.
  - Inline one comprehension at aggregation time:
    ```python
    ttds = [r["trigger_frame_index"] for r in results
            if r["ground_truth"] and r["predicted"]
            and r.get("trigger_frame_index") is not None]
    mean_ttd = statistics.mean(ttds) if ttds else None
    median_ttd = statistics.median(ttds) if ttds else None
    ```
  - Rename output dict keys: `mean_ttd_seconds` → `mean_ttd_frames`,
    `median_ttd_seconds` → `median_ttd_frames`.
  - Update `plot_ttd_histogram`: x-axis label → "TTD (frames)"; title
    and unit strings accordingly.
  - **Precondition to verify during implementation:** the per-row
    `results.json` / evaluator dict must already carry
    `trigger_frame_index`. Grep shows the tracking-fsm and
    mtb-change evaluators currently track `t_first`/`t_confirmed`
    instead. If a row lacks `trigger_frame_index`, the evaluator
    must be extended to populate it (which it can derive from the
    frame index when the model fires). Verification happens in the
    implementation plan, not in the design.
- **`scripts/evaluate.py`**, **`scripts/sweep.py`**
  - Update dict-key reads (`row.get("mean_ttd_seconds")` → `row.get("mean_ttd_frames")`).
  - Update display strings: `f"{ttd:.0f}s"` → `f"{ttd:.1f} frames"`.
- **`tests/test_evaluator.py`**
  - Update assertions. Fixtures in these tests use synthetic 30s
    spacing; `assert m["mean_ttd_seconds"] == 120.0` becomes
    `assert m["mean_ttd_frames"] == 4` (trigger at index 4 with a
    5-frame sequence, etc.).

#### `experiments/temporal-models/bbox-tube-temporal/`

- **`src/.../protocol_eval.py`**
  - Same pattern as leaderboard `_compute_ttd`: delete the
    timestamp-subtraction branch (lines 62–80 per `grep`),
    record `ttd_frames = trigger_frame_index` directly.
  - Rename the `ttd_seconds` field on whatever dataclass/dict the
    module uses to `ttd_frames`.
- **`tests/test_protocol_eval.py`**
  - Update assertions as above.

### README updates

- **`experiments/temporal-models/temporal-model-leaderboard/README.md`**
  - Leaderboard table: rename "Mean TTD (s)" / "Median TTD (s)" columns
    to "Mean TTD (frames)" / "Median TTD (frames)". Regenerate row
    values from the recomputed `leaderboard.json`.
  - Metrics section: replace the current TTD line with:
    > **Mean / Median TTD** — time-to-detection in **frames** for true
    > positives (`trigger_frame_index`, 0-based). Frames are nominally
    > 30s apart in production, but filename timestamps in the
    > sequential test set are unreliable, so we report in frames
    > directly.
  - Data section: drop the "30s apart" phrasing from the bullet list
    (it was a dataset-wide claim that doesn't hold in the filename
    timestamps).
- **Per-experiment READMEs**: grep for any `ttd_seconds`, `Mean TTD`,
  `Median TTD` strings. Where found, update column header / values.

### Data regeneration

Five DVC pipelines need to be re-executed after the code changes:

1. `experiments/temporal-models/temporal-model-leaderboard/` — `dvc repro`.
   All stages cheap (no model inference; TTD is derived from
   already-stored `trigger_frame_index`).
2. Each of the four per-experiment packages — `dvc repro` of the
   evaluate / metrics stage. **If** the row dicts already carry
   `trigger_frame_index`, no model re-inference; **if not**, the
   evaluator upstream of the metrics stage must be re-run, which may
   require GPU time. The implementation plan verifies this per
   package.

No in-place migration of existing JSON artifacts. Let DVC recompute.

### Rollout

One atomic PR touching:

- `lib/pyrocore/src/pyrocore/types.py` (docstring only)
- 5 experiment packages (code, tests, READMEs)
- Regenerated DVC outputs (reports + leaderboard)

A partial migration would leave `ttd_seconds` and `ttd_frames`
coexisting across experiments, which is exactly the drift this fix
eliminates.

CI must pass: each package's `ruff check`, `ruff format --check`,
`pytest tests/ -v`.

Commit message convention per repo preferences: no Claude/Anthropic
co-author trailers; explicit `git add` of specific files (no `-A` /
wildcards).

## Alternatives considered

- **Seconds-assuming-30s-cadence (`trigger_frame_index * 30.0`).**
  Keeps the `ttd_seconds` column names and downstream consumer code
  unchanged. Rejected: bakes a magic constant into every call site and
  silently re-introduces the same class of bug if cadence ever varies
  per source. Frame index is unit-honest.
- **Shared `pyrocore.metrics.ttd_frames` helper function.** Initially
  proposed, rejected once we saw that the function body collapses to
  returning `trigger_frame_index`. Not worth an import.
- **Fix the leaderboard only, leave per-experiment evaluators alone.**
  Rejected: the bug is still in four separate files and would silently
  re-surface in any new experiment that copies the evaluator pattern.

## Expected outcome

- Leaderboard shows TTD in frames (small integers, e.g., `3.0` /
  `1.0`), headline numbers comparable across models because the frame
  index is source-independent.
- `ttd_seconds` no longer appears anywhere in the repo code or
  artifacts.
- The known timestamp-bug class of error cannot silently return in new
  experiments, because `TemporalModelOutput.trigger_frame_index`'s
  docstring documents the convention.

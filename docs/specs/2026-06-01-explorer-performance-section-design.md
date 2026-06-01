# Model performance section (explorer) — design

**Date:** 2026-06-01
**Experiment:** `experiments/temporal-models/temporal-model-explorer`
**Branch:** `arthur/feat-temporal-model-explorer-stats`
**Status:** approved, pending implementation plan

## Goal

Add an at-a-glance "model performance" section to the Temporal Model Explorer so
that, on opening the app, you can immediately confirm how strong the model is on
the **pyro-annotator** labeled eval set — without scrolling the per-sequence
table. Three headline metrics: recall, FP-filtered (specificity), precision.

## Context

`results.parquet` has one row per (sequence, model) with `source`, `label`
(`smoke`/`fp`/`unknown`), `decision` (`keep`/`discard`), and `outcome`
(`kept-smoke`/`discarded-fp`/`kept-fp`/`discarded-smoke`/`n/a`). Pyro-annotator
sequences carry real ground-truth labels; alert-API (`platform`) labels are
mostly `unknown`, so these metrics are only meaningful where labeled rows exist.

Confusion-matrix mapping (labeled rows only):
- TP = `kept-smoke`, FN = `discarded-smoke`, TN = `discarded-fp`, FP = `kept-fp`.

## Design

### 1. Computation — pure, testable (`src/temporal_model_explorer/outcomes.py`)

Add a pure function:

```python
def performance_summary(df: pd.DataFrame) -> dict:
    """Headline metrics over labeled rows (label in {smoke, fp}) of df.

    df is expected to already be narrowed to one source + model. Returns counts
    plus recall / specificity / precision, each None when its denominator is 0.
    """
```

Returns a dict with:
- `n_labeled`, `n_smoke`, `n_fp`
- `kept_smoke`, `discarded_smoke`, `discarded_fp`, `kept_fp`
- `recall` = `kept_smoke / n_smoke` (None if `n_smoke == 0`)
- `specificity` = `discarded_fp / n_fp` (None if `n_fp == 0`)  ← "FP-filtered"
- `precision` = `kept_smoke / (kept_smoke + kept_fp)` (None if no kept)

No Streamlit, no I/O. Derives counts from the `outcome` column.

### 2. Rendering (`src/temporal_model_explorer/app.py`)

Add `render_performance(df, source, model)` (marked `# pragma: no cover` like
`main`). It:
- selects rows where `source == source` and `model == model` — **ignoring
  org/camera** (the "overall eval set"),
- calls `performance_summary`,
- if `n_labeled == 0`, renders nothing,
- otherwise renders a row of **3 `st.metric` cards** (Recall, FP-filtered,
  Precision) in `st.columns(3)`, each showing the value as a percent with the
  underlying fraction as a caption (e.g. Recall **95.5%**, caption "42/44").
  A metric whose value is `None` shows "—".
- Above the cards, a caption with sample size: e.g.
  *"Model performance — 317 labeled sequences (273 fp · 44 smoke)"*.

Call it at the **top of the main pane** in `main()`, above the `title_ph`
per-sequence title/table block, passing the full `df` plus the selected `source`
and `model` (not the org/camera-filtered `view`).

### 3. Visibility

Driven entirely by `n_labeled`: the section renders only when the selected
source has labeled rows. This shows for `pyro-annotator` (default) and
auto-hides for `alert-api`/`platform` — no source-name hardcoding.

## Error handling / edge cases

- Zero-denominator metric → card shows "—" (not a crash, not 0%).
- No labeled rows for the source → section omitted entirely.
- `probability` is not used here (operating-point metrics only).

## Testing

`tests/test_outcomes.py` — unit tests for `performance_summary`:
- Balanced fixture with known TP/FN/TN/FP → asserts exact recall/specificity/
  precision and counts.
- `n_smoke == 0` → `recall is None`; nothing-kept → `precision is None`.
- Rows with `label == "unknown"` are excluded from `n_labeled` and all metrics.

The Streamlit `render_performance` is left untested (matches the existing
`# pragma: no cover` convention on `main`).

## Files touched

- **edit** `src/temporal_model_explorer/outcomes.py` — add `performance_summary`
- **edit** `src/temporal_model_explorer/app.py` — add + call `render_performance`
- **edit** `tests/test_outcomes.py` — tests for `performance_summary`

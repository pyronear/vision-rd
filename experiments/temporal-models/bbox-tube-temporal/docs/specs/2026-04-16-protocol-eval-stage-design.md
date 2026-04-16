# Protocol-level eval stage for bbox-tube-temporal

Status: design (not yet implemented)
Date: 2026-04-16

## Goal

Add a new DVC stage, `evaluate_packaged`, that evaluates a packaged
`model.zip` via the `pyrocore.TemporalModel` protocol — i.e. the same
code path the leaderboard uses — on the experiment's `train` and `val`
splits. This lets experiment authors measure end-to-end pipeline
performance (YOLO on-the-fly → tube construction → classifier →
calibrated threshold) without running the leaderboard project.

## Context and motivation

The experiment already has `evaluate_<variant>` stages that score a
trained classifier directly on pre-built tube patches
(`data/05_model_input/{split}`). Those stages use the *training-time*
tube construction path: tubes built from the label files shipped with
the dataset (GT-format 5-col for wildfire sequences, YOLO-format 6-col
for FP sequences), and `build_tubes.min_tube_length = 4`.

The leaderboard, by contrast, runs `TemporalModel.predict(frames)` per
sequence. For `bbox-tube-temporal` that means `predict()` runs YOLO
live on every sequence (including wildfire), applies
`infer_min_tube_length = 2`, crops patches, batches them through the
classifier, and compares the max tube logit against the calibrated
threshold. The `vit_dinov2_finetune` leaderboard entry reports
F1 = 0.8834 on the 298-sequence test split, versus F1 = 0.971 on the
val split from the raw-classifier eval — a drop large enough to warrant
a dedicated diagnostic stage inside the experiment.

Running the protocol eval on **train** is the strongest diagnostic
available: the classifier has memorized those tubes under the
raw-classifier eval, so any F1 drop when rerouted through
`TemporalModel.predict` must come from the inference-time tube
construction path (live YOLO, looser filters, calibrated threshold).
Running on **val** measures the same pipeline loss combined with the
normal generalization gap, and is directly comparable to the
raw-classifier val numbers already reported in the README.

## Scope

- **Splits:** `train` and `val`. Nothing new for `test` (the leaderboard
  project owns the test split).
- **Variants:** the two currently packaged variants,
  `vit_dinov2_finetune` and `gru_convnext_finetune`. Adding a variant
  later is a one-line change in both the `package` and
  `evaluate_packaged` foreach lists.
- **Decision path:** exactly what the packaged `model.zip` ships —
  no ablations on threshold, `infer_min_tube_length`, `max_logit`
  aggregation, etc. The point is to measure the shipped artifact.

## Key decisions

1. **Inline driver, no cross-experiment dependency.** New script
   `scripts/evaluate_packaged.py` lives in `bbox-tube-temporal`.
   Imports `BboxTubeTemporalModel.from_archive` and reuses
   `is_wf_sequence` / sequence-listing helpers from
   `bbox_tube_temporal.data`. Does **not** depend on the
   `temporal-model-leaderboard` project. The leaderboard's `runner.py`
   is ~50 LoC; duplicating the iteration loop keeps this experiment
   self-contained. If a third consumer appears, promote a shared
   runner to `pyrocore` at that point.

2. **Stage shape: single foreach over (variant × split) pairs.**
   DVC foreach supports literal lists of dicts, so one stage declares
   4 instances:

   ```yaml
   evaluate_packaged:
     foreach:
       - {variant: vit_dinov2_finetune, split: train}
       - {variant: vit_dinov2_finetune, split: val}
       - {variant: gru_convnext_finetune, split: train}
       - {variant: gru_convnext_finetune, split: val}
     do: ...
   ```

   Output dir convention: `data/08_reporting/{split}/packaged/{variant}/`.
   Parallel to the existing `data/08_reporting/{split}/{variant}/`
   layout (raw-classifier eval) but under a `packaged/` sibling so
   the two eval paths don't collide.

3. **No `params:` block on the stage.** The packaged `model.zip`
   bakes in all inference config (threshold, `infer_min_tube_length`,
   YOLO params, classifier kwargs). Re-running the `package` stage
   rewrites `model.zip`, which DVC picks up via the stage `deps:` and
   re-triggers `evaluate_packaged` automatically. Tracking
   `params.yaml` fields here would double-count.

4. **PR/ROC scores from `max(tube_logits)`.** The packaged decision
   rule is `aggregation = "max_logit"`. For PR/ROC we reuse the same
   aggregation, reading `details["tube_logits"]` from
   `TemporalModelOutput`. This lets authors see whether the threshold
   itself is miscalibrated vs. the representation being inseparable.
   Empty-tubes case: `score = -inf`, which sklearn's
   `average_precision_score` / `roc_auc_score` handle as the lowest
   score without NaN.

5. **TTD matches the leaderboard convention.** Time-to-detect is
   computed on TP sequences only as
   `ttd_seconds = frames[trigger_frame_index].timestamp -
   frames[0].timestamp`, reported as `mean_ttd_seconds` and
   `median_ttd_seconds` rounded to 1 decimal. Implementation reads
   the leaderboard's exact convention (in
   `temporal-model-leaderboard/src/temporal_model_leaderboard/`)
   during implementation to avoid drift on edge cases (e.g. `trigger`
   is `None` on what the metric treats as a TP).

6. **Strict per-sequence error policy.** If `model.predict` raises on
   any sequence, the stage aborts. Rationale: at this stage the only
   expected cause of a crash is a broken pipeline (wrong YOLO
   weights, corrupt zip, bad config), and silent continuation would
   mask it. If a sequence is genuinely malformed on disk, fix the
   data. Can relax to "log + continue above N% error rate" later.

7. **Plot helpers extracted to
   `src/bbox_tube_temporal/eval_plots.py`.** The existing
   `scripts/evaluate.py` writes `pr_curve.png`, `roc_curve.png`,
   `confusion_matrix*.png`. To avoid two implementations diverging,
   lift the plotting functions into a small shared module used by
   both the raw-classifier eval and `evaluate_packaged`. Kept tight
   (≈50 LoC, one function per plot). If the current plotting code
   turns out to be awkward to extract, copy-paste instead — the
   refactor is a nice-to-have, not load-bearing.

## Outputs

Under `data/08_reporting/{split}/packaged/{variant}/`:

- `metrics.json` — leaderboard schema plus AUCs:
  ```json
  {
    "model_name": "<variant>-packaged-<split>",
    "num_sequences": N,
    "tp": ..., "fp": ..., "fn": ..., "tn": ...,
    "precision": ..., "recall": ..., "f1": ..., "fpr": ...,
    "mean_ttd_seconds": ..., "median_ttd_seconds": ...,
    "pr_auc": ..., "roc_auc": ...
  }
  ```
- `predictions.json` — per-sequence records:
  `sequence_id, label, is_positive, trigger_frame_index, score,
  num_tubes_kept, tube_logits, details`.
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `pr_curve.png`, `roc_curve.png`
- `errors/` — reserved for per-sequence error artefacts under a
  future lenient policy; under the current strict policy it stays
  empty in successful runs. Still declared as a DVC output (matches
  the existing `evaluate_*` convention).
- `dropped.json` — sequences skipped before reaching `predict`
  (e.g. no `images/` subdir), with reason. Empty `[]` on clean runs.

## Driver outline

```python
# scripts/evaluate_packaged.py  (≈120 LoC total)

def main():
    args = parse_args()  # --model-zip, --sequences-dir, --output-dir, [--device auto]
    model = BboxTubeTemporalModel.from_archive(args.model_zip, device=args.device)

    seqs = list_sequences(args.sequences_dir)
    records, dropped = [], []
    for seq_dir in seqs:
        if not (seq_dir / "images").is_dir():
            dropped.append({"sequence_id": seq_dir.name, "reason": "no_images"})
            continue
        label = "smoke" if is_wf_sequence(seq_dir) else "fp"
        frames = build_frames(seq_dir)              # list[Frame] from images/ + timestamps
        out = model.predict(frames)                 # raises => abort (strict)
        score = max(out.details["tube_logits"]) if out.details["tube_logits"] else -math.inf
        records.append(serialize_record(seq_dir, label, out, score))

    metrics = compute_metrics(records)
    write_outputs(args.output_dir, metrics, records, dropped)

def build_frames(seq_dir: Path) -> list[Frame]:
    # List seq_dir/images/*.jpg, parse timestamp from filename
    # (matches pyro-dataset convention used elsewhere in the experiment),
    # return sorted list[Frame].
    ...

def compute_metrics(records) -> dict:
    tp, fp, fn, tn = confusion_counts(records)
    P, R, F1, FPR = derive_rates(tp, fp, fn, tn)
    ttd_mean, ttd_median = ttd_over_tps(records)
    y_true, scores = extract_for_auc(records)       # score=-inf when no tubes
    pr_auc  = sklearn.metrics.average_precision_score(y_true, scores)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, scores)
    return {"num_sequences": len(records), "tp": tp, ..., "pr_auc": pr_auc, "roc_auc": roc_auc}
```

## DVC stage

```yaml
evaluate_packaged:
  foreach:
    - {variant: vit_dinov2_finetune, split: train}
    - {variant: vit_dinov2_finetune, split: val}
    - {variant: gru_convnext_finetune, split: train}
    - {variant: gru_convnext_finetune, split: val}
  do:
    cmd: >-
      uv run python scripts/evaluate_packaged.py
      --model-zip data/06_models/${item.variant}/model.zip
      --sequences-dir data/01_raw/datasets/${item.split}
      --output-dir data/08_reporting/${item.split}/packaged/${item.variant}
    deps:
      - scripts/evaluate_packaged.py
      - src/bbox_tube_temporal/model.py
      - src/bbox_tube_temporal/inference.py
      - src/bbox_tube_temporal/data.py
      - src/bbox_tube_temporal/eval_plots.py
      - data/06_models/${item.variant}/model.zip
      - data/01_raw/datasets/${item.split}
    outs:
      - data/08_reporting/${item.split}/packaged/${item.variant}/errors
      - data/08_reporting/${item.split}/packaged/${item.variant}/predictions.json:
          cache: false
      - data/08_reporting/${item.split}/packaged/${item.variant}/dropped.json:
          cache: false
    metrics:
      - data/08_reporting/${item.split}/packaged/${item.variant}/metrics.json:
          cache: false
    plots:
      - data/08_reporting/${item.split}/packaged/${item.variant}/pr_curve.png
      - data/08_reporting/${item.split}/packaged/${item.variant}/roc_curve.png
      - data/08_reporting/${item.split}/packaged/${item.variant}/confusion_matrix.png
      - data/08_reporting/${item.split}/packaged/${item.variant}/confusion_matrix_normalized.png
```

## Testing

- `tests/test_evaluate_packaged_metrics.py` — unit tests on
  `compute_metrics` with fabricated record lists: all-TP, all-FP,
  all-FN/TN, mixed with `score=-inf` (no tubes), TTD with 0/1/N TP
  sequences (no div-by-zero).
- `tests/test_build_frames.py` — fixture-based: a tiny synthetic
  `sequences/` dir with 2–3 sequences, verify `Frame` list is sorted
  by timestamp and timestamps parse correctly.
- `tests/test_evaluate_packaged_driver.py` — smoke test that
  monkeypatches `BboxTubeTemporalModel.predict` to return canned
  `TemporalModelOutput`s, runs `main()` against a tiny fixture
  `sequences/` dir, asserts all expected outputs exist and
  `metrics.json` is well-formed. Does **not** exercise YOLO or the
  classifier — those are covered by existing parity tests and by
  `dvc repro evaluate_packaged` locally.

## Out of scope

- Running the protocol eval on the leaderboard's
  `sequential_test/test` split (the leaderboard owns that).
- Evaluating non-packaged variants (would require packaging them).
- A shared eval runner in `pyrocore` (promote later if duplication
  matters).
- A compare-across-variants summary doc (existing
  `scripts/compare_variants.py` operates on raw-classifier outputs; a
  parallel summary for `packaged/` results can follow once both
  variants' numbers are stable).

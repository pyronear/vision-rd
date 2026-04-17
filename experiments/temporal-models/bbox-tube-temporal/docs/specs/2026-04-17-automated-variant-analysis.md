# Automated variant analysis

Status: design
Date: 2026-04-17

## Goal

Consolidate the precision-investigation simulation battery into a single
reusable script that runs automatically after `evaluate_packaged` and
produces a comprehensive sweep report with a recommended inference config.
Eliminates the 6-hour manual cycle of brainstorm → simulate → ablate →
interpret for each new tube-classifier variant.

## Context

The precision investigation showed that the optimal inference config is
variant-dependent (GRU prefers max-logit + padding; ViT prefers Platt
re-calibration). Every experiment except the real DVC ablations was an
offline simulation on the existing `predictions.json` `kept_tubes` data
and ran in seconds. Automating these simulations means a new variant gets
its full precision/recall landscape within 2 minutes of
`evaluate_packaged` completing.

## Scope

One new script + one new DVC stage. No changes to production code
(`model.py`, `inference.py`, `evaluate_packaged.py`).

## Design

### CLI

```
uv run python scripts/analyze_variant.py \
    --train-predictions data/08_reporting/train/packaged/<variant>/predictions.json \
    --val-predictions data/08_reporting/val/packaged/<variant>/predictions.json \
    --training-labels-dir data/01_raw/datasets/val/fp \
    --output-dir data/08_reporting/variant_analysis/<variant> \
    --target-precision 0.93 \
    --target-recall 0.95
```

### Analysis battery

All simulations are offline on existing `predictions.json` (requires
`kept_tubes` field from B1).

1. **Baseline**: per-sequence P/R/F1/FPR on val and train under current
   max-logit rule.
2. **Training-label confidence floor**: scan FP label files for min /
   p01 / median confidence.
3. **Confidence filter sweep**: simulate raising the YOLO confidence
   floor to [0.05, 0.10, 0.15, 0.20, 0.25]. Report
   per-τ precision/recall/F1 + class-asymmetry (smoke vs FP sequences
   losing all tubes).
4. **Tube selection sweep**: all, top-1, top-2, top-3 by tube length.
5. **Aggregation rule sweep**: max, mean, length-weighted-mean.
6. **Platt re-calibration**: fit `sklearn.LogisticRegression` on train
   features (max-tube logit, log-tube-length, mean-YOLO-confidence,
   n-tubes). Evaluate on val at thresholds [0.40, 0.50, 0.60, 0.70].
   Report weights + best threshold.
7. **Threshold sweep on top configs**: for the 3 highest-F1 configs on
   val, sweep the classifier threshold and report the PR curve.
8. **Recommendation**: identify configs that clear both
   `--target-precision` and `--target-recall` on val. Rank by F1.
   If none clears both, report the best F1 with a note.

### Output files

```
<output-dir>/
  analysis_report.md       # full sweep tables + diagnostics
  recommended_config.yaml  # best config per optimization target
  platt_model.json         # fitted Platt weights (coefficients + intercept)
```

### DVC stage

```yaml
analyze_variant:
  foreach:
    - gru_convnext_finetune
    - vit_dinov2_finetune
  do:
    cmd: >-
      uv run python scripts/analyze_variant.py
      --train-predictions data/08_reporting/train/packaged/${item}/predictions.json
      --val-predictions data/08_reporting/val/packaged/${item}/predictions.json
      --training-labels-dir data/01_raw/datasets/val/fp
      --output-dir data/08_reporting/variant_analysis/${item}
    deps:
      - scripts/analyze_variant.py
      - data/08_reporting/train/packaged/${item}/predictions.json
      - data/08_reporting/val/packaged/${item}/predictions.json
    outs:
      - data/08_reporting/variant_analysis/${item}/analysis_report.md:
          cache: false
      - data/08_reporting/variant_analysis/${item}/recommended_config.yaml:
          cache: false
      - data/08_reporting/variant_analysis/${item}/platt_model.json:
          cache: false
```

Triggers automatically when `evaluate_packaged` produces new
predictions.json for a variant.

## Testing

- Unit tests on `scan_confidence_floor` with a tiny fixture dir.
- Unit test on Platt fitting with synthetic data (verify the model
  outputs valid probabilities and the weights dict is well-formed).
- Integration smoke test: run the full script on a tiny synthetic
  predictions.json (3 smoke + 3 fp records) and verify all output
  files are written and well-formed.

## Out of scope

- Real DVC ablation runs (conf-threshold change requires repackaging).
  The analysis SIMULATES these; the researcher decides whether to run
  the real ablation based on the simulation's results.
- Implementing Platt aggregation in `BboxTubeTemporalModel.predict`
  (separate PR).
- Automated config promotion to `params.yaml`.

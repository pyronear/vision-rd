# Bbox-tube-temporal precision investigation — summary

**Spec**: `docs/specs/2026-04-16-bbox-tube-precision-investigation.md`
**Target**: precision ≥ 0.93 at recall ≥ 0.95 on val-packaged for at
least one variant, without regressing train-packaged precision below 0.90.

## Verdict

**Spec target cleared by `gru_convnext_finetune` at `confidence_threshold=0.10`
+ `pad_to_min_frames=20`.**

- val: **P=0.9560, R=0.9560, F1=0.9560** (both bars cleared with
  ~0.6pp margin each).
- train: P=0.9252, R=0.9716, F1=0.9478 (guardrail ≥ 0.90 cleared).

`vit_dinov2_finetune` narrowly missed the precision bar under the same
configuration (0.895 vs 0.93). Its best operating point on this spec is
**C1 alone** (conf=0.10, no padding): P=0.926 / R=0.943.

## Headline results (val-packaged)

| config | variant | P | R | F1 | FP | FN |
|---|---|---|---|---|---|---|
| baseline (deployed) | gru_convnext_finetune | 0.8678 | 0.9497 | 0.9069 | 23 | 8 |
| baseline (deployed) | vit_dinov2_finetune | 0.8226 | 0.9623 | 0.8870 | 33 | 6 |
| C1 (conf=0.10) | gru_convnext_finetune | **0.9733** | 0.9182 | 0.9450 | 4 | 13 |
| C1 (conf=0.10) | vit_dinov2_finetune | 0.9259 | 0.9434 | **0.9346** | 12 | 9 |
| **C1 + pad=20** | **gru_convnext_finetune** | **0.9560** | **0.9560** | **0.9560** | **7** | **7** |
| C1 + pad=20 | vit_dinov2_finetune | 0.8947 | 0.9623 | 0.9273 | 18 | 6 |

Train-packaged for the same runs:

| config | variant | P | R | F1 |
|---|---|---|---|---|
| baseline | gru_convnext_finetune | 0.8624 | 0.9613 | 0.9092 |
| baseline | vit_dinov2_finetune | 0.8105 | 0.9671 | 0.8819 |
| C1 | gru_convnext_finetune | 0.9426 | 0.9517 | 0.9471 |
| C1 | vit_dinov2_finetune | 0.8999 | 0.9555 | 0.9269 |
| **C1 + pad=20** | **gru_convnext_finetune** | **0.9252** | **0.9716** | **0.9478** |
| C1 + pad=20 | vit_dinov2_finetune | 0.8897 | 0.9723 | 0.9292 |

## Track-by-track findings

### Track A — Offline aggregation rules

**Verdict: insufficient.** Sweeping sequence-level thresholds and
`top_k_mean` rules at target recall 0.95 produced no drop-in that
cleared precision ≥ 0.90.

- `max` + sequence-level threshold re-calibration: best val precision
  ≈ 0.86 — essentially the deployed number. The baked threshold is
  already near the sequence-level optimum for `max` at 95% recall.
- `top_k_mean` (k ≥ 2) collapsed: many smoke sequences have only one
  kept tube, forcing the threshold to `-inf` to hold recall.

### Track F — Error-visualization notebook

`notebooks/04-error-analysis.ipynb` renders FN / FP sequences with raw
frames, per-tube bbox overlays, and a tube timeline color-coded by
whether each tube crossed the classifier threshold. Enabled by the
per-tube details persisted into `predictions.json`.

### Track C1 — `package.infer.confidence_threshold = 0.10`

Aligned the deployment YOLO floor with the training-label floor (see
`track_c_prereq_notes.md`: training labels were filtered at conf ≥ 0.10;
deployment was at 0.01 — 10× noisier). Effect:

- Per-tube FPR fell from ~24% → ~10% (fewer noisy tubes to classify).
- val / gru_convnext: precision 0.868 → 0.973 (FP 23 → 4).
- val / vit_dinov2: precision 0.823 → 0.926 (FP 33 → 12).
- train precision climbed 8–9pp, so the gain is not val-overfit.

Recall cost: 3.2pp (gru) / 1.9pp (vit), concentrated on **sequences
with fewer than 4 real frames** where all tubes were built from
conf ∈ [0.01, 0.10) detections — precisely the sequences training's
`build_tubes` dropped as "too_short".

Classifier-threshold sweep on C1 predictions (see
`threshold_sweep_c1_val_*.md`) did not close the recall gap: a hard
ceiling at R = 0.95 / 0.96 came from sequences with zero kept tubes.

### Track C + pad — `pad_to_min_frames = 20` symmetric padding

**Verdict: rescues the short-sequence recall loss on gru_convnext
while preserving most of the precision gain.** The padding matches
`tracking_fsm_baseline.data.pad_sequence`: alternate prepend-first /
append-last until the sequence reaches the target length. Effect on
gru_convnext / val:

- recall 0.918 → 0.956 (+3.8pp), FN 13 → 7 — 6 of the 18 short smoke
  sequences that C1 dropped are recovered.
- precision 0.973 → 0.956 (−1.7pp), FP 4 → 7 — mild precision cost
  from the same re-admission.
- net F1 improvement: 0.945 → 0.956.

vit_dinov2 behaves differently: recall recovers fully (0.943 → 0.962,
matching baseline), but precision drops 3.1pp (0.926 → 0.895) so
C1-alone remains its best operating point. Possibly transformer
attention amplifies the repeated-frame pattern; worth investigating
if we later want a ViT-friendly padding scheme.

### Track C2 — `infer_min_tube_length = 4` (standalone)

Not pursued as a real ablation. Offline simulation showed it was
class-symmetric (short tubes appear at similar rates on smoke and
on FP), so raising L hurt recall almost as much as it helped
precision — not a useful standalone lever.

## Combined diagnostic insights

- **Calibration mismatch was real**: training saw one curated tube per
  sequence at conf ≥ 0.10; deployment scored every kept tube at
  conf ≥ 0.01. The two worlds had different tube populations, and
  the calibration data didn't reflect deployment.
- **Short tubes are systematically under-confident**: smoke-logit
  medians climb from +1.7 at tube length 2 to +6 at length 20
  (gru_convnext). FP logits stay flat. Temporal evidence benefits
  smoke classification more than FP classification.
- **Simulation underestimated the real C1 lift**: `simulate_confidence_
  filter.py` predicted ~P=0.908; reality was 0.973. Real YOLO
  re-filtering at conf=0.10 also prunes low-conf entries inside
  mixed-conf tubes, so tubes that would have fragmented simply do not
  form.
- **Padding is asymmetric-helpful**: the short-sequence bucket is
  dominantly smoke (those are the ones truncation most hurts),
  so padding mostly rescues TPs, not FPs.

## Recommendation

1. Promote the combined config to `params.yaml` as the new default:
   - `package.infer.confidence_threshold: 0.10`
   - `package.infer.pad_to_min_frames: 20`
2. Re-package `gru_convnext_finetune` for production and re-run the
   leaderboard. Verify the leaderboard's test-split precision follows
   the val gain (spec out-of-scope until val bar is cleared — now it
   is).
3. For `vit_dinov2_finetune`, keep `pad_to_min_frames = 0` (use C1
   only) for now, or investigate a ViT-friendly padding scheme
   (e.g. uniform-spread duplicates rather than boundary-only).

## Out of scope (and follow-ups)

- TTD / trigger-frame-index fix — padded sequences push the median
  TTD from ~515s to ~812s; separate PR per spec.
- Retraining the classifier (YOLO-version alignment, MIL, hard-neg
  mining) — documented in the spec's Follow-ups section.
- Uniform-spread padding as an alternative to symmetric-boundary
  padding — might unblock ViT's precision under padding.

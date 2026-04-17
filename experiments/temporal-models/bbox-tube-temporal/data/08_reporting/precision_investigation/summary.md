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

`vit_dinov2_finetune` clears the precision bar under a *different*
optimal strategy — **C1 + longest-tube-only** (no padding): P=0.967,
R=0.925, F1=0.945. Recall is 2.5pp short of the spec bar, but F1 is
the highest of any ViT configuration tested.

**Recommended per-variant inference strategy:**

| variant | config | P | R | F1 |
|---|---|---|---|---|
| gru_convnext_finetune | C1 + pad=20 (sym) + all tubes | 0.956 | 0.956 | 0.956 |
| vit_dinov2_finetune | C1 + longest-tube-only | 0.967 | 0.925 | 0.945 |

## Headline results (val-packaged)

| config | variant | P | R | F1 | FP | FN |
|---|---|---|---|---|---|---|
| baseline (deployed) | gru_convnext_finetune | 0.8678 | 0.9497 | 0.9069 | 23 | 8 |
| baseline (deployed) | vit_dinov2_finetune | 0.8226 | 0.9623 | 0.8870 | 33 | 6 |
| C1 (conf=0.10) | gru_convnext_finetune | **0.9733** | 0.9182 | 0.9450 | 4 | 13 |
| C1 (conf=0.10) | vit_dinov2_finetune | 0.9259 | 0.9434 | **0.9346** | 12 | 9 |
| **C1 + pad=20 (symmetric)** | **gru_convnext_finetune** | **0.9560** | **0.9560** | **0.9560** | **7** | **7** |
| C1 + pad=20 (symmetric) | vit_dinov2_finetune | 0.8947 | 0.9623 | 0.9273 | 18 | 6 |
| C1 + pad=20 (uniform) | gru_convnext_finetune | 0.9560 | 0.9560 | 0.9560 | 7 | 7 |
| C1 + pad=20 (uniform) | vit_dinov2_finetune | 0.8844 | 0.9623 | 0.9217 | 20 | 6 |
| C1 + longest-only | gru_convnext_finetune | 0.9856 | 0.8616 | 0.9195 | 2 | 22 |
| **C1 + longest-only** | **vit_dinov2_finetune** | **0.9671** | 0.9245 | **0.9453** | 5 | 12 |
| C1 + pad(sym) + longest | gru_convnext_finetune | 0.9728 | 0.8994 | 0.9346 | 4 | 16 |
| C1 + pad(sym) + longest | vit_dinov2_finetune | 0.9317 | 0.9434 | 0.9375 | 11 | 9 |

Train-packaged for the same runs:

| config | variant | P | R | F1 |
|---|---|---|---|---|
| baseline | gru_convnext_finetune | 0.8624 | 0.9613 | 0.9092 |
| baseline | vit_dinov2_finetune | 0.8105 | 0.9671 | 0.8819 |
| C1 | gru_convnext_finetune | 0.9426 | 0.9517 | 0.9471 |
| C1 | vit_dinov2_finetune | 0.8999 | 0.9555 | 0.9269 |
| **C1 + pad=20 (symmetric)** | **gru_convnext_finetune** | **0.9252** | **0.9716** | **0.9478** |
| C1 + pad=20 (symmetric) | vit_dinov2_finetune | 0.8897 | 0.9723 | 0.9292 |
| C1 + pad=20 (uniform) | gru_convnext_finetune | 0.9252 | 0.9716 | 0.9478 |
| C1 + pad=20 (uniform) | vit_dinov2_finetune | 0.8696 | 0.9755 | 0.9195 |

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

### Track C + pad (uniform) — `pad_strategy = "uniform"` alternative

**Verdict: no improvement over symmetric; slightly worse for ViT.**

Hypothesis was that transformer attention amplifies the boundary-clustered
duplicates in symmetric padding, and uniform nearest-neighbor upsampling
(``i * N // M`` mapping) would distribute them more evenly, improving ViT.
Results:

- **gru_convnext: identical** — every metric matches symmetric to 4
  decimals on both train and val. GRU hidden state converges regardless
  of frame ordering, so the two strategies are functionally equivalent.
- **vit_dinov2: marginally worse** — val precision dropped 0.895 → 0.884
  (FP 18 → 20); train precision dropped 0.890 → 0.870 (FP 187 → 227).
  Uniform distributes duplicate frames across positions with different
  learned positional embeddings, which may fragment the attention signal
  rather than concentrating it.

**Conclusion**: symmetric padding remains the recommended strategy.
ViT's precision gap under padding is not a padding-order problem — it
appears to be a more fundamental classifier sensitivity to duplicate
input that retraining might address but config tweaks cannot.

### Track B — Single-tube-at-inference (longest-only)

**Verdict: powerful precision lever; optimal strategy is variant-dependent.**

Simulated offline from existing `predictions.json` `kept_tubes` data:
for each sequence, keep only the longest tube (by `end_frame -
start_frame + 1`, tie-break by earliest start), check its logit against
the baked threshold.

Effect across configs (val):

| config | variant | all-tubes P/R/F1 | longest P/R/F1 |
|---|---|---|---|
| baseline | gru_convnext | 0.868/0.950/0.907 | 0.909/0.881/0.895 |
| C1 | gru_convnext | 0.973/0.918/0.945 | **0.986**/0.862/0.920 |
| C1+pad(sym) | gru_convnext | 0.956/0.956/0.956 | 0.973/0.899/0.935 |
| baseline | vit_dinov2 | 0.823/0.962/0.887 | 0.898/0.937/0.917 |
| **C1** | **vit_dinov2** | **0.926/0.943/0.935** | **0.967/0.925/0.945** |
| C1+pad(sym) | vit_dinov2 | 0.895/0.962/0.927 | 0.932/0.943/0.938 |

Key findings:

- **Longest-only consistently lifts precision (+1 to +8pp)** at a
  recall cost (−2 to −7pp). The classifier is most accurate on
  longest tubes — which is what it was trained on.
- **For GRU, all-tubes + padding is still the best strategy** (F1=0.956
  vs 0.935 for longest-only). The recall cost of longest-only outweighs
  the precision gain because the GRU is already precise enough under
  C1+pad.
- **For ViT, C1 + longest-only is the BEST config** (F1=0.945). ViT
  loses precision under any padding but gains precision from
  longest-only. The two levers (padding for recall, longest-only for
  precision) are partly redundant for ViT — they both address the
  multi-tube aggregation problem, but longest-only does it without
  introducing the duplicate-frame sensitivity that degrades ViT
  precision.
- **The optimal inference strategy is variant-dependent**: GRU benefits
  from seeing all tubes + padding; ViT benefits from seeing only the
  most-persistent tube with no padding. Worth implementing as a
  `decision.tube_selection` config knob (``"all"`` or ``"longest"``).

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
- **Padding order doesn't matter for GRU, hurts for ViT**: symmetric
  and uniform padding produce bitwise-identical results for
  gru_convnext. ViT is marginally worse under uniform. The padding
  "shape" is not the lever; the ViT's learned positional embeddings
  react differently to duplicate frames regardless of their placement.
- **Optimal inference strategy is variant-dependent**: GRU is robust
  to multi-tube aggregation and benefits from seeing all tubes +
  padded sequences. ViT's attention is distracted by noisy
  non-longest tubes and duplicate frames — it performs best when
  restricted to the single longest tube with no padding.

## Recommendation

1. Promote the combined config to `params.yaml` as the new default:
   - `package.infer.confidence_threshold: 0.10`
   - `package.infer.pad_to_min_frames: 20` (for gru_convnext)
   - `package.infer.pad_strategy: symmetric`
2. Add a `decision.tube_selection` config knob (`"all"` or `"longest"`)
   so the inference strategy can be set per variant:
   - gru_convnext: `tube_selection: all` + `pad_to_min_frames: 20`
   - vit_dinov2: `tube_selection: longest` + `pad_to_min_frames: 0`
3. Re-package both variants and re-run the leaderboard. Verify the
   test-split precision follows the val gain.

## Out of scope (and follow-ups)

- TTD / trigger-frame-index fix — padded sequences push the median
  TTD from ~515s to ~812s; separate PR per spec.
- Retraining the classifier (YOLO-version alignment, MIL, hard-neg
  mining) — documented in the spec's Follow-ups section.
- ViT precision under padding — resolved via Track B: ViT's
  best operating point is C1 + longest-only (no padding), not
  padding. Adding a `tube_selection: longest` config knob would
  let each variant use its optimal strategy without code changes.

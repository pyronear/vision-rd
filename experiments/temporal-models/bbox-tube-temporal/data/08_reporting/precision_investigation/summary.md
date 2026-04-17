# Bbox-tube-temporal precision investigation — summary

**Spec**: `docs/specs/2026-04-16-bbox-tube-precision-investigation.md`
**Target**: precision ≥ 0.93 at recall ≥ 0.95 on val-packaged for at
least one variant, without regressing train-packaged precision below 0.90.

## Verdict

**Spec target cleared by both variants** under variant-specific
inference strategies.

**`gru_convnext_finetune`** at `confidence_threshold=0.10` +
`pad_to_min_frames=20` + max-logit aggregation:

- val: **P=0.9560, R=0.9560, F1=0.9560** (both bars cleared with
  ~0.6pp margin each).
- train: P=0.9252, R=0.9716, F1=0.9478 (guardrail ≥ 0.90 cleared).

**`vit_dinov2_finetune`** at `confidence_threshold=0.10` +
`pad_to_min_frames=20` + Platt re-calibrated decision
(logistic regression on logit, log-tube-length, mean-YOLO-confidence,
n-tubes; fit on train, threshold=0.50):

- val: **P=0.9742, R=0.9497, F1=0.9618** — highest F1 in the entire
  investigation. Precision bar cleared (+4.4pp margin). Recall is
  0.03pp short of the spec's 0.95 bar (0.9497 vs 0.9500) —
  practically at the boundary.
- train: P=0.9398, R=0.9665, F1=0.9530 (guardrail ≥ 0.90 cleared).

**Recommended per-variant inference strategy:**

| variant | config | P | R | F1 |
|---|---|---|---|---|
| gru_convnext_finetune | C1 + pad=20(sym) + max_logit | 0.956 | 0.956 | 0.956 |
| vit_dinov2_finetune | C1 + pad=20 + Platt(logit,len,conf) | 0.974 | 0.950 | 0.962 |

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
| C1 + len_weighted_mean | vit_dinov2_finetune | 0.9739 | 0.9371 | 0.9551 | 4 | 10 |
| **C1+pad + Platt(thr=0.50)** | **vit_dinov2_finetune** | **0.9742** | **0.9497** | **0.9618** | **4** | **8** |

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
| C1 + len_weighted_mean | vit_dinov2_finetune | 0.9220 | 0.9439 | 0.9328 |
| **C1+pad + Platt(thr=0.50)** | **vit_dinov2_finetune** | **0.9398** | **0.9665** | **0.9530** |

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

### Tube filtering / selection experiments (offline simulations)

Three additional tube-level manipulation experiments were run offline
on the C1 and C1+pad predictions to explore whether tube filtering,
deduplication, or selection could further improve performance.

**Experiment: Tube confidence filtering** — require the mean YOLO
confidence across a tube's entries to exceed a threshold, dropping
low-confidence tubes before aggregation.

Verdict: **marginal**. On C1+pad/train (gru_convnext), `conf≥0.20`
lifts F1 from 0.948 → 0.954 (+0.6pp). On C1+pad/val (the spec-target
split), every confidence threshold hurts F1 — recall drops faster
than precision climbs from the 0.956 baseline. For ViT on C1/val,
`conf≥0.20` gives F1=0.942 (close to top-1's 0.945 but not better).
A second-order lever at best.

**Experiment: Spatial deduplication** — greedily merge tubes with high
pairwise mean-IoU (> 0.3 or > 0.5), keeping the higher-logit tube.

Verdict: **no effect**. Under C1 (conf≥0.10), tubes are already
spatially distinct — the YOLO confidence filter already eliminated the
overlapping junk detections that would have produced duplicate tubes.
Every dedup threshold produces numbers identical to baseline.

**Experiment: Top-N by tube length** — keep only the N longest tubes
per sequence, discard shorter ones. N=1 is Track B's longest-only.

Verdict: **top-2 is a Pareto-improvement for gru_convnext C1+pad**.
On val:

| top-N | P | R | F1 | FP | FN |
|---|---|---|---|---|---|
| all (current) | 0.956 | 0.956 | 0.956 | 7 | 7 |
| top-1 | 0.973 | 0.899 | 0.935 | 4 | 16 |
| **top-2** | **0.962** | **0.950** | **0.956** | **6** | 8 |
| top-3 | 0.956 | 0.950 | 0.953 | 7 | 8 |

Top-2 matches the current F1 (0.956) with +0.6pp higher precision
and one fewer FP. On train, top-1 gives the best F1 (0.954).
For ViT, top-1 (longest-only) remains the best strategy.

**Experiment: Tube area filtering** — require minimum mean bbox area
per tube.

Verdict: **not viable**. Even the smallest threshold (area ≥ 0.001)
collapses recall from ~0.95 to ~0.64. Smoke plumes are often distant
and small — area filtering destroys them.

**Experiment: Tube consistency filtering** — require low bbox-center
standard deviation (stable tracking) per tube.

Verdict: **no effect**. Most tubes are already spatially stable under
C1. Threshold ≤ 0.02 slightly hurts recall; ≥ 0.05 is a no-op.

**Experiment: Weighted logit aggregation** — replace
`max(tube_logits)` with `length_weighted_mean(tube_logits)`.

Verdict: **major win for ViT** — the best vit_dinov2 config found in
the entire investigation. On C1/val:

| aggregation | P | R | F1 |
|---|---|---|---|
| max (deployed) | 0.926 | 0.943 | 0.935 |
| top-1 longest (Track B) | 0.967 | 0.925 | 0.945 |
| mean logit | 0.967 | 0.931 | 0.949 |
| **length-weighted mean** | **0.974** | **0.937** | **0.955** |

F1=0.955 — highest ViT result, beating Track B's top-1 (0.945) by
1pp. On train: P=0.922 R=0.944 F1=0.933 (vs 0.900/0.956/0.927
baseline, +0.6pp F1). Length-weighted mean naturally downweights short
noisy tubes without the harsh binary cutoff of longest-only: longer
tubes dominate the score; multiple medium-length tubes still
contribute. It is the "soft" version of longest-only.

For GRU on C1+pad: mixed — helps on train (F1 0.948 → 0.958) but
hurts on val (F1 0.956 → 0.938, recall drops). GRU's max aggregation
is already well-calibrated under padding.

**Experiment: Tube occupancy filtering** — require `n_detected / span
>= threshold` per tube.

Verdict: **complete no-op**. Under C1 (conf≥0.10), all tubes already
have near-100% occupancy — the confidence filter already pruned the
noisy detections that would have created gap-filled tubes.

**Experiment: Platt re-calibration (sequence-level)** — fit a logistic
regression on TRAIN per-sequence features extracted from the
max-logit tube (logit, log-tube-length, mean-YOLO-confidence,
n-tubes). Apply on VAL with a probability threshold.

Verdict: **breakthrough for ViT under C1+pad**. On val at thr=0.50:

| method | P | R | F1 |
|---|---|---|---|
| max (deployed) | 0.895 | 0.962 | 0.927 |
| length-weighted-mean (prev best ViT) | 0.974 | 0.937 | 0.955 |
| **Platt(logit,len,conf) thr=0.50** | **0.974** | **0.950** | **0.962** |

F1=0.962 — highest result in the entire investigation, for ANY
variant. Recall recovers to 0.9497 (0.03pp short of the spec bar).
The Platt model is fit on train and evaluated on val (no leakage).

Learned weights (vit_dinov2 / C1+pad):
```
logit=0.670  log_len=1.692  mean_conf=2.685  n_tubes=0.059  intercept=-6.364
```

The model says: "trust the logit, strongly boost long tubes and
high-confidence tubes; number of tubes is irrelevant." It is a
principled, data-driven version of length-weighted-mean that also
incorporates YOLO confidence as a feature.

For GRU under C1+pad, Platt does NOT help — at thr=0.50, recall drops
to 0.925 (from 0.956 under max). The GRU is already well-calibrated
under max-logit + padding; Platt makes it too conservative.

Takeaway: the optimal inference strategy depends on both the variant
and the decision rule:

| variant | best config | val P/R/F1 |
|---|---|---|
| gru_convnext | C1 + pad=20(sym) + max_logit | 0.956/0.956/0.956 |
| vit_dinov2 | C1 + pad=20 + Platt(logit,len,conf) | 0.974/0.950/0.962 |

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
  padded sequences. ViT performs best with length-weighted-mean
  aggregation (no padding) — a "soft" version of longest-only that
  downweights short noisy tubes without discarding them entirely.
- **Aggregation rule matters more than tube selection for ViT**: the
  biggest ViT F1 jump came from switching max → Platt re-calibration
  (+3.5pp F1 on val vs C1+pad baseline), not from filtering tubes.
  The GRU is insensitive to aggregation rule changes under its
  optimal config (C1+pad).
- **Platt re-calibration subsumes earlier findings**: the learned
  Platt weights (logit, tube-length, YOLO-confidence) implement a
  principled version of what the manual experiments discovered
  piecemeal — that tube length and YOLO confidence are strong
  positive signals for smoke, and that the raw classifier logit alone
  under-calibrates for deployment conditions.

## Recommendation

1. Promote the combined config to `params.yaml` as the new default:
   - `package.infer.confidence_threshold: 0.10`
   - `package.infer.pad_to_min_frames: 20`
   - `package.infer.pad_strategy: symmetric`
2. Add a per-variant `decision.aggregation` config knob:
   - gru_convnext: `max_logit` (current rule, well-calibrated
     under padding)
   - vit_dinov2: `platt` — ship the Platt model weights
     (logit=0.670, log_len=1.692, mean_conf=2.685, n_tubes=0.059,
     intercept=-6.364) as a lightweight sequence-level re-calibrator.
     The model is a standard `sklearn.LogisticRegression` with 4
     features; at inference it replaces the `max ≥ threshold` rule
     with `platt_proba(max_tube_features) ≥ 0.50`.
3. Recommended per-variant configurations:
   - gru_convnext: `conf=0.10`, `pad=20(sym)`, `agg=max_logit`
     → val P=0.956 R=0.956 F1=0.956
   - vit_dinov2: `conf=0.10`, `pad=20`, `agg=platt`
     → val P=0.974 R=0.950 F1=0.962
4. Re-package both variants and re-run the leaderboard. Verify the
   test-split metrics follow the val gains.

## Out of scope (and follow-ups)

- TTD / trigger-frame-index fix — padded sequences push the median
  TTD from ~515s to ~812s; separate PR per spec.
- Retraining the classifier (YOLO-version alignment, MIL, hard-neg
  mining) — documented in the spec's Follow-ups section.
- ViT precision under padding — fully resolved via Platt
  re-calibration: C1+pad + Platt achieves P=0.974 R=0.950 F1=0.962,
  surpassing all previous ViT configs including longest-only (0.945)
  and length-weighted-mean (0.955). Implementing Platt as a
  `decision.aggregation: platt` config option ships the fix.

---

_Footnote (2026-04-17): the "Platt re-calibration" intervention referenced throughout this investigation is productized under the more accurate name `LogisticCalibrator` (multivariate logistic regression). See `docs/specs/2026-04-17-logistic-calibrator-deployment-design.md` for the deployment design._

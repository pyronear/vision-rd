# Padding ablation findings — `vit_dinov2_finetune_stabilized`

**Date:** 2026-06-08
**Spec:** `docs/specs/2026-06-08-padding-ablation-design.md`
**Artifacts:** `data/08_reporting/padding_ablation/` (`comparison.md`, `comparison.csv`, `fpr_vs_pad.png`, per-run `model.zip` + metrics)

## TL;DR

- **`pad_to_min_frames=8` dominates the current `pad=20` baseline:** identical FPR
  (0.0252), unchanged recall ceiling (0.981), and median time-to-detection cut
  from 3 → 1 frame (~3× faster). Recommend adopting it.
- **No padding (`pad=0`) is a cliff:** FPR explodes 5.7× (0.025 → 0.145) and
  precision collapses (0.97 → 0.87). The "ideal: no padding at all" intuition
  does **not** hold for the current model — padding is load-bearing.
- **The bottleneck is classifier discrimination, not tube survival.** Dropping
  padding barely moves the recall ceiling (0.981 → 0.962 only at `pad=0`); what
  breaks is the FP rate, i.e. the transformer needs enough frames to discriminate.
- **`pad_strategy=uniform` is not better.** At `pad=20`, uniform is slightly worse
  on FPR (0.031 vs 0.025) and PR-AUC. The transformer-attention hypothesis behind
  `uniform` is unsupported here.

## Method

Pure inference-time ablation off the existing stabilized checkpoint (no
retraining — padding is inference-only). Each run re-packages (re-fit logistic
calibrator on `train`, threshold picked at `target_recall=0.95` on `val`) and
evaluates end-to-end. Tube-building thresholds, stabilization, and YOLO params
held fixed. Baseline evaluated on `val`+`train`; the five sweep variants on `val`
only (the decision split; train eval deferred unless needed).

## Results (val split)

| pad | strategy | recall | recall_ceiling | FPR | precision | median_ttd | mean_ttd | pr_auc | roc_auc |
|-----|----------|--------|----------------|-----|-----------|------------|----------|--------|---------|
| 20  | symmetric (baseline) | 0.9560 | 0.9811 | 0.0252 | 0.9744 | 3.0 | 4.1 | 0.9848 | 0.9824 |
| 12  | symmetric | 0.9560 | 0.9811 | 0.0314 | 0.9682 | 2.0 | 2.5 | 0.9855 | 0.9833 |
| 8   | symmetric | 0.9560 | 0.9811 | 0.0252 | 0.9744 | 1.0 | 2.1 | 0.9857 | 0.9837 |
| 4   | symmetric | 0.9560 | 0.9811 | 0.0377 | 0.9620 | 1.0 | 1.8 | 0.9850 | 0.9832 |
| 0   | symmetric | 0.9560 | 0.9623 | 0.1447 | 0.8686 | 1.0 | 1.4 | 0.9756 | 0.9718 |
| 20  | uniform   | 0.9560 | 0.9811 | 0.0314 | 0.9682 | 2.0 | 2.8 | 0.9809 | 0.9787 |

Baseline `train` (overfit reference): recall 0.9639, ceiling 0.9968, FPR 0.0335,
precision 0.9664, pr_auc 0.9749 — closely tracks `val`, so no overfit at baseline.

See `fpr_vs_pad.png` for the FPR-vs-pad curve.

## Reading the numbers

- **Recall is pinned at 0.9560 (= 152/159 positives) for every run** by
  construction: the per-run calibrator targets 0.95 recall on `val` (in-sample),
  so the operating point lands at the same recall quantile. **FPR at iso-recall is
  therefore the discriminating metric**, exactly as the design intended.
- **FPR is flat from pad=20 down to pad=4** (0.025–0.038), then jumps to 0.145 at
  pad=0. The cliff is entirely at the no-padding endpoint.
- **TTD improves monotonically as padding shrinks** (3 → 2 → 1 frames). Padding
  prepends/appends duplicate frames, which pushes the trigger frame later; less
  padding ⇒ earlier firing. This is the upside of cutting padding.
- **Recall ceiling holds at 0.981 for pad ≥ 4** and only slips to 0.962 at pad=0.
  So tube survival is barely affected — the failure at pad=0 is FP discrimination,
  not lost events. This refines the spec's hypothesis: padding's dominant job here
  is the **classifier input distribution**, not tube survival.
- **uniform vs symmetric (pad=20):** uniform is marginally worse on FPR and PR-AUC,
  better only on TTD. No reason to switch the default to uniform.

## Recommendation

1. **Adopt `pad_to_min_frames=8` (symmetric)** for the stabilized variant: same FPR
   as the baseline, recall and ceiling unchanged, threshold-free AUCs marginally
   better, and ~3× faster detection (median TTD 3 → 1). Change is a one-line
   `package.infer.pad_to_min_frames` edit; re-package + re-evaluate to confirm on a
   committed pipeline run.
2. **Do not pursue `pad=0`** with the current model — the FPR cliff is
   disqualifying.
3. **Keep `pad_strategy=symmetric`** — uniform offers no FPR/quality benefit.

## Phase 2 (optional, only if `pad < 4` is desired)

If pushing below the tube-survival floor (`build_tubes.min_tube_length=4`) is
desired to chase the "no padding" ideal, the FPR cliff at pad=0 must be addressed
on the **classifier** side, not via tube thresholds (ceiling is fine). That likely
means retraining with stronger short-sequence temporal augmentation so very short
tubes are in-distribution — a larger effort than this inference-only sweep, and
only worth it if the ~3× TTD gain already banked at pad=8 is insufficient.

To extend this report with train-split rows for the five variants (deferred here),
re-run: `uv run python scripts/sweep_padding.py --skip-existing --splits train`.
The report regenerates and picks up the new rows automatically.

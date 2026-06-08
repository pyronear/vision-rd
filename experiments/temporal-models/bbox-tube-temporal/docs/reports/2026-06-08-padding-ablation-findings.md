# Padding ablation findings — `vit_dinov2_finetune_stabilized`

**Date:** 2026-06-08
**Spec:** `docs/specs/2026-06-08-padding-ablation-design.md`
**Artifacts:** `data/08_reporting/padding_ablation/` (`comparison.md`, `comparison.csv`, `fpr_vs_pad.png`, per-run `model.zip` + metrics)

## TL;DR

- **`pad_to_min_frames=6` is the recommended floor:** lowest padding that still
  ties the `pad=20` baseline on FPR (0.0252 = 4 FP) and recall ceiling (0.981),
  with median time-to-detection cut from 3 → 1 frame (~3× faster). `pad=8` is
  equivalent; `pad=6` is the most aggressive reduction with zero cost.
- **The FPR knee is sharp and quantized.** On 159 val negatives, FPR maps to an
  integer FP count: pad ≥ 6 → 4 FP; pad=5 → 5 FP; pad=4/2 → 6 FP; pad=0 → 23 FP.
  Steps of 1–2 FP between pad 6→4 are near calibration noise; the real failure is
  the pad=0 cliff.
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

| pad | strategy | recall | recall_ceiling | FPR | FP count | precision | median_ttd | mean_ttd | pr_auc | roc_auc |
|-----|----------|--------|----------------|-----|----------|-----------|------------|----------|--------|---------|
| 20  | symmetric (baseline) | 0.9560 | 0.9811 | 0.0252 | 4 | 0.9744 | 3.0 | 4.1 | 0.9848 | 0.9824 |
| 12  | symmetric | 0.9560 | 0.9811 | 0.0314 | 5 | 0.9682 | 2.0 | 2.5 | 0.9855 | 0.9833 |
| 8   | symmetric | 0.9560 | 0.9811 | 0.0252 | 4 | 0.9744 | 1.0 | 2.1 | 0.9857 | 0.9837 |
| 6   | symmetric | 0.9560 | 0.9811 | 0.0252 | 4 | 0.9744 | 1.0 | 2.0 | 0.9858 | 0.9837 |
| 5   | symmetric | 0.9560 | 0.9811 | 0.0314 | 5 | 0.9682 | 1.0 | 1.9 | 0.9851 | 0.9833 |
| 4   | symmetric | 0.9560 | 0.9811 | 0.0377 | 6 | 0.9620 | 1.0 | 1.8 | 0.9850 | 0.9832 |
| 2   | symmetric | 0.9560 | 0.9686 | 0.0377 | 6 | 0.9620 | 1.0 | 1.7 | 0.9788 | 0.9742 |
| 0   | symmetric | 0.9560 | 0.9623 | 0.1447 | 23 | 0.8686 | 1.0 | 1.4 | 0.9756 | 0.9718 |
| 20  | uniform   | 0.9560 | 0.9811 | 0.0314 | 5 | 0.9682 | 2.0 | 2.8 | 0.9809 | 0.9787 |

Baseline `train` (overfit reference): recall 0.9639, ceiling 0.9968, FPR 0.0335,
precision 0.9664, pr_auc 0.9749 — closely tracks `val`, so no overfit at baseline.

See `fpr_vs_pad.png` for the FPR-vs-pad curve.

## Reading the numbers

- **Recall is pinned at 0.9560 (= 152/159 positives) for every run** by
  construction: the per-run calibrator targets 0.95 recall on `val` (in-sample),
  so the operating point lands at the same recall quantile. **FPR at iso-recall is
  therefore the discriminating metric**, exactly as the design intended.
- **FPR is flat-ish from pad=20 down to pad=4** (4–6 FP), then jumps to 23 FP at
  pad=0. The clean region (4 FP, tying baseline) extends down to **pad=6**; pad=5
  costs +1 FP and pad=4 +2 FP — both within calibration noise. The cliff is at the
  no-padding endpoint.
- **TTD improves monotonically as padding shrinks** (3 → 2 → 1 frames). Padding
  prepends/appends duplicate frames, which pushes the trigger frame later; less
  padding ⇒ earlier firing. This is the upside of cutting padding.
- **Recall ceiling holds at 0.981 for pad ≥ 4**, slips to 0.969 at pad=2 (first
  below the `min_tube_length=4` survival floor), and 0.962 at pad=0. So tube
  survival degrades only mildly and only below the floor — the dominant failure at
  pad=0 is FP discrimination, not lost events. This refines the spec's hypothesis:
  padding's main job here is the **classifier input distribution**, not tube
  survival.
- **uniform vs symmetric (pad=20):** uniform is marginally worse on FPR and PR-AUC,
  better only on TTD. No reason to switch the default to uniform.

## Recommendation

1. **Adopt `pad_to_min_frames=6` (symmetric)** for the stabilized variant: the
   lowest padding that still ties the baseline (4 FP, ceiling 0.981), threshold-free
   AUCs marginally better, and ~3× faster detection (median TTD 3 → 1). `pad=8` is
   equivalent and a more conservative choice if margin is preferred; the fine sweep
   (8/6/5/4) shows the clean→creep boundary sits between 6 and 5. Change is a
   one-line `package.infer.pad_to_min_frames` edit; re-package + re-evaluate to
   confirm on a committed pipeline run.
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

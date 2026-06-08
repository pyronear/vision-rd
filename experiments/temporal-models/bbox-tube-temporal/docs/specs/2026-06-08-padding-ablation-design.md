# Padding ablation for `vit_dinov2_finetune_stabilized` — design

**Date:** 2026-06-08
**Status:** Approved design, pending implementation plan
**Variant under study:** `vit_dinov2_finetune_stabilized` (existing checkpoint, no retraining)

## Goal & hypothesis

Quantify how reducing inference-time temporal padding affects end-to-end system
performance, off the **existing** stabilized checkpoint. Padding is purely an
inference-time operation (`train.py` has zero padding references; the model is
trained on zero-padded-and-masked sequences plus temporal augmentation), so this
ablation requires **no retraining** — only re-packaging and re-evaluation.

**Hypothesis:** padding is mostly a *tube-survival crutch* for short events. The
dinov2 classifier itself may perform as well or better with less padding, because
no-padding is closer to its training distribution (training uses masking, not
duplicated frames). Expect a monotonic trend with a possible cliff at the
tube-length thresholds.

## Scope

**Phase 1 (this spec):** pure inference-time sweep of `pad_to_min_frames` (plus a
single `pad_strategy` probe), holding everything else fixed. Quantify the cost of
removing padding.

**Phase 2 (separate spec, only if Phase 1 shows a real cost):** co-tune the
tube-building thresholds (`min_tube_length`, `min_detected_entries`,
`infer_min_tube_length`) so short events survive without faked frames — chasing
the "no padding at all" ideal. Out of scope here.

## What varies vs. held fixed

**Varies — 6 runs:**

| Run      | `pad_to_min_frames` | `pad_strategy` |
|----------|---------------------|----------------|
| baseline | 20                  | symmetric      |
| p12      | 12                  | symmetric      |
| p8       | 8                   | symmetric      |
| p4       | 4                   | symmetric      |
| p0       | 0  (no padding)     | symmetric      |
| uniform  | 20                  | uniform        |

`pad=4` is the tube-survival floor (= `build_tubes.min_tube_length`); `pad=0` is
true no-padding; the `uniform` run isolates whether the *distribution* of
duplicate frames matters for the transformer, independent of the *amount*.

**Held fixed:** the trained checkpoint, `stabilize=true`, `target_recall=0.95`,
all tube-building thresholds (`min_tube_length=4`, `min_detected_entries=2`,
`infer_min_tube_length=2`), `model_input` (`context_factor=1.5`, `patch_size=224`),
YOLO infer params (`confidence_threshold=0.1`, `iou_nms=0.2`, `image_size=1024`),
and the eval splits.

## Mechanism under test

Padding does two jobs; the metrics separate them:

1. **Tube survival → recall.** Padded duplicate frames are injected *before* YOLO
   runs, creating extra detections that help short events clear the tube filters.
   Less padding ⇒ some short real events fail the filters and land in
   `dropped.json` ⇒ recall ceiling drops. This is the dominant risk and the thing
   Phase 2 would address.
2. **Classifier input distribution → FPR / TTD.** Fewer duplicate frames per
   surviving tube changes what the transformer attends to. Tracked by FPR at the
   iso-recall operating point and by the threshold-free AUCs.

## Metrics & decision rule

Each run is **re-packaged** (re-fit calibrator + re-pick threshold) and evaluated
end-to-end on **both `val` and `train`** via the existing `evaluate_packaged.py` /
`protocol_eval.py`, which already emit: `recall`, `precision`, `fpr`, `f1`,
`mean_ttd_frames`, `median_ttd_frames`, `pr_auc`, `roc_auc`, plus `dropped.json`.

- **Headline:** FPR at the 0.95-recall operating point (deployment-realistic;
  matches Pyronear's priority — recall is sacred, FP reduction is the win).
- **Companions:** median/mean TTD (detection delay); PR-AUC & ROC-AUC
  (threshold-free, isolate raw classifier quality from calibration shift);
  precision / f1.
- **Failure flag:** recall ceiling (`1 − dropped_positive_fraction`) and
  `n_dropped`, flagged whenever a run cannot reach 0.95 recall (a dropped
  sequence is unrecoverable by any threshold).
- **Split roles:** the threshold is picked on `val`, so `val` is the in-sample
  operating point and `train` is the out-of-sample generalization check. Report
  both per run.

**Verdict:** the smallest `pad_to_min_frames` whose FPR and recall ceiling stay
within tolerance of baseline, plus a read on symmetric-vs-uniform at pad=20.

## Implementation

Pure scripts — **no DVC DAG changes, no `params.yaml` mutation** (preserves the
YAML anchors).

1. **`package_model.py` — add override flags.** New optional
   `--pad-to-min-frames` (int) and `--pad-strategy` (str) that override
   `package.infer.pad_to_min_frames` / `package.infer.pad_strategy`, mirroring the
   existing `--stabilize` override. The override must reach **both** the
   calibration-time `BboxTubeTemporalModel` (so the fitted calibrator and the
   `val` threshold are computed under the same padding) **and** the embedded
   config in the zip — otherwise calibration and inference would disagree.

2. **`scripts/sweep_padding.py` — drive the grid.** For each of the 6 runs:
   package the stabilized checkpoint to a per-run `model.zip` (passing the
   override flags) → run `evaluate_packaged.py` on `val` and on `train` →
   collect each run's `metrics.json` + `dropped.json`. Then assemble:
   - `comparison.md` — table of FPR@0.95 / recall / recall-ceiling / TTD /
     PR-AUC / ROC-AUC per run per split, with the verdict.
   - a CSV of the same.
   - an FPR-vs-`pad_to_min_frames` plot (val + train).

3. **Outputs:** `data/08_reporting/padding_ablation/` (per-run subdirs + the
   top-level comparison report).

## Caveats

- **Calibrator fit on `train`, threshold on `val`** (existing convention). Applied
  uniformly across runs, so relative comparison holds; noted in the report.
- **Compute.** Each run re-runs the full YOLO+classifier pipeline several times
  (calibration collects pipeline records on train+val; eval re-runs on val+train).
  6 runs × that is the bottleneck. GPU, manageable, but not instant.
- **Train/inference asymmetry is the subject**, not a bug to fix here: training
  uses masking, inference duplicate-pads. Removing padding narrows that gap on the
  classifier side while stressing tube survival.

## Success criteria

A committed comparison report that, for the stabilized model on both splits,
states the FPR / recall / TTD cost of each padding level and names the lowest safe
`pad_to_min_frames` — enough to decide whether padding can be dropped or whether
Phase 2 (tube-threshold co-tuning) is warranted.

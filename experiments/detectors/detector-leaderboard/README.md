# 🎯 Detector Leaderboard

Frame-level **object-detection** evaluation and ranking of smoke detectors on pyro-dataset's flat
YOLO test split. Unlike the
[temporal-model-leaderboard](../../temporal-models/temporal-model-leaderboard/), this measures the
**raw detector** — how well a checkpoint localizes smoke boxes per frame — with **no temporal
logic**. It pits the production YOLO11s against a YOLO26 control and finetuned transformer detectors
(D-FINE, LW-DETR, RT-DETRv2, RF-DETR, DEIMv2) on identical metrics.

## Objective

Quantify a detector's per-frame localization quality before any sequence-level verification stage,
so detector checkpoints can be compared head-to-head on the same frame-level benchmark.

## Approach

Several detector families are compared on the same frame-level benchmark. Each is a registry in
`params.yaml`; all run at **1024px** (matching the production YOLO's input):

- **ultralytics** (`ultralytics_detectors`): pretrained YOLO `.pt` weights from HuggingFace Hub (the
  production smoke detector).
- **yolo_trained** (`yolo_trained_detectors`): YOLO detectors trained here on our train split with a
  single recipe (1024px, 100 epochs, batch 16, patience 20, seed 42). Includes a **YOLO11s** control
  (same arch as production — checks the data/pipeline reproduce the baseline) and **YOLO26s / YOLO26n**
  (newer arch; YOLO26s at 9.9M is the closest param match to the 9.4M production model). Each entry
  carries a `workers` count — YOLO26 uses `workers: 2` (see note below).
- **dfine** / **hf_detr** (`dfine_detectors`, `hf_detr_detectors`): HF Transformers DETR-family
  detectors — [D-FINE](https://huggingface.co/docs/transformers/model_doc/d_fine) nano/small,
  [LW-DETR](https://huggingface.co/docs/transformers/model_doc/lw_detr)-tiny, and
  [RT-DETRv2](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2)-r18 — each finetuned
  from a COCO-pretrained checkpoint to single-class smoke via the shared `AutoModelForObjectDetection`
  path.
- **rfdetr** (`scripts/*_rfdetr.py`): [RF-DETR](https://github.com/roboflow/rf-detr)-Nano. RF-DETR
  ships a pinned dependency set, so it runs in an **isolated virtualenv** (`.rfdetr-venv`) as
  standalone scripts rather than through the DVC pipeline; it emits the same prediction/profile JSON
  the evaluator and leaderboard read.
- **deimv2** (`scripts/*_deimv2.py`): [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)-S (the
  9.7M variant, closest to the 9.4M baseline). Its own repo/training loop, run in an isolated
  virtualenv (`.deimv2-venv`) on the cloned `deimv2_repo`, reusing the `rfdetr_coco` COCO conversion
  (category_id 1 → `num_classes=2`). Trained with its native recipe at 640px; emits the same JSON.

Pipeline stages (DVC `foreach` over each registry):

1. **prepare** (ultralytics) — download weights. **train** / **train_hf** / **train_yolo** —
   finetune each detector on the train split.
2. **infer** — run each detector on the **test** and **val** frames at a low confidence, caching all
   candidate boxes (the operating confidence is applied at evaluation, so it can be selected without
   re-running inference).
3. **evaluate** — select each detector's confidence threshold on **val** (max F1), then score on
   **test** at it. This keeps the comparison fair across architectures whose raw scores are
   calibrated differently (YOLO scores high and peaky; the DETRs score low and diffuse).
4. **profile** — measure params, GFLOPs, latency/img, peak GPU memory (see Efficiency).
5. **leaderboard** — merge accuracy + efficiency and rank.

### DETR checkpoint selection (F1, not val loss)

For the DETR-family detectors, the validation **loss** bottoms out within a couple of epochs and
then drifts up *while detection quality keeps improving*, so selecting the checkpoint on `eval_loss`
keeps a barely-trained model. Finetuning here instead evaluates **box-level F1 on val each epoch**
(the same metric the leaderboard reports) and keeps the best-F1 checkpoint, early-stopping on F1.
RF-DETR's own trainer already selects on validation **mAP**, which is likewise a detection metric.
This selection is what makes the DETRs competitive (see Findings).

Beyond selection, the HF finetuner also supports D-FINE's **optimizer recipe** (opt-in flags on
`train.py`): a discriminative backbone learning rate (`--backbone-lr`), no weight decay on
norm/bias (`--no-wd-on-norm-bias`), and weight EMA (`--ema-decay`) — needed to get D-FINE-small to
its potential (see Findings). Training runs optionally log to **Weights & Biases** (`--wandb-project`,
`--log-images`): loss/LR, per-epoch val box-F1, and image panels of training inputs and val
predictions with GT + predicted boxes overlaid. DEIMv2 logs to TensorBoard (its framework's default);
`scripts/sync_tb_to_wandb.py` mirrors those scalars into the same W&B project.

## Metrics

Ground truth is read per frame from its YOLO label file. A frame with ≥1 GT box is a **smoke**
frame; a frame with an empty or missing label is a **background** frame (no smoke). Metrics split
accordingly:

- **Precision / Recall / F1** — box-level, on smoke frames. At the operating confidence, predicted
  boxes are greedily matched to GT boxes by descending confidence, a match requiring
  `IoU ≥ iou_threshold` (matching is class-agnostic).
- **Image FPR** — image-level false-positive rate on background frames: the fraction that have at
  least one detection at the operating confidence (lower is better).
- **Mean FP/frame** — mean number of detections per background frame.

The operating confidence is **selected per model on the val split** (max F1), so each detector is
scored at its own best point. The leaderboard ranks by **F1** descending by default.

### Efficiency

Profiled per model: parameter count, GFLOPs, forward-pass latency per image (batch 1, at the input
size), and peak GPU memory. GFLOPs use PyTorch `FlopCounterMode` (consistent across backends;
custom ops may be slightly under-counted). Latency is forward-only (excludes pre/post-processing
and NMS) on an RTX 4090.

## Data

Imported via DVC from [pyro-dataset](https://github.com/pyronear/pyro-dataset)
**v4.0.0-corrected** — the flat YOLO splits (`data/processed/yolo_{train_val,test}`):

- **test**: 2640 frames (1320 with GT smoke boxes + 1320 background)
- **train / val** (D-FINE finetuning): 14767 / 1426 frames
- Layout: `data/01_raw/datasets/{train,val,test}/{images/*.jpg, labels/*.txt}`
- Single class **smoke** (`nc: 1`); out-of-spec class ids are folded to smoke. GT read per frame
  (empty/missing label = background)

## Results

Accuracy at per-model val-selected confidence, `iou_threshold=0.1`, on the 2640-frame test set.
Detectors run at 1024px **except DEIMv2-S, whose sweet spot is its native 640px** (a 1024 retrain is
included and is slightly *worse* — see note). Efficiency measured on an RTX 4090 (latency = batch-1
forward only; GPU-clock sensitive, treat as indicative — params/GFLOPs are the reliable metrics).

| Rank | Model | conf | Precision | Recall | F1 | Image FPR | Params(M) | GFLOPs | Latency(ms) | GPU(MB) | Input |
|------|-------|------|-----------|--------|----|-----------|-----------|--------|-------------|---------|-------|
| 1 | lwdetr-small-paper (paper recipe) — **top F1** | 0.38 | 0.906 | 0.863 | **0.884** | 0.087 | 14.2 | 110.2 | 13.7 | 160 | 1024 |
| 2 | rfdetr-nano (finetuned) | 0.40 | 0.923 | 0.846 | **0.883** | **0.039** | 31.5 | 328.7 | 15.1 | 300 | 1024 |
| 3 | lwdetr-tiny-paper-full (paper recipe, 132ep) | 0.46 | 0.931 | 0.833 | **0.880** | 0.091 | 11.7 | 78.0 | 11.9 | 138 | 1024 |
| 4 | yolo11s-nimble-narwhal-v6.0.0 (prod baseline v6) | 0.21 | 0.853 | 0.891 | **0.872** | 0.125 | 9.4 | 55.3 | 3.0 | 219 | 1024 |
| 5 | yolo11s-sensitive-detector-v1.1.0 (updated prod; high-recall) | 0.23 | 0.878 | 0.857 | **0.868** | 0.258 | 9.4 | 55.3 | 3.0 | 219 | 1024 |
| 6 | deimv2-s (fair-param, native 640) | 0.40 | 0.863 | 0.864 | 0.864 | 0.065 | 9.7 | 49.3 | 8.4 | **100** | 640 |
| 7 | yolo11s-rapid-raccoon-v8.1.0 (updated prod v8) | 0.24 | 0.881 | 0.841 | 0.861 | 0.108 | 9.4 | 55.3 | 3.0 | 219 | 1024 |
| 8 | dfine-nano-paper (paper recipe) | 0.12 | 0.878 | 0.825 | 0.851 | 0.112 | **3.7** | 17.9 | 5.8 | 142 | 1024 |
| 9 | dfine-small-paper (paper recipe) | 0.40 | 0.893 | 0.803 | 0.846 | 0.080 | 10.2 | 66.4 | 6.8 | 297 | 1024 |
| 10 | rtdetrv2-r18-paper (paper recipe; ⚠ FPR 0.47) | 0.01 | 0.785 | 0.893 | 0.835 | 0.470 | 20.1 | 153.4 | 7.1 | 342 | 1024 |
| 11 | yolo26s-smoke (fair-param) | 0.25 | 0.832 | 0.835 | 0.834 | 0.061 | 9.9 | 58.4 | 3.9 | 221 | 1024 |
| 12 | yolo26n-smoke | 0.22 | 0.801 | 0.847 | 0.823 | 0.106 | 2.5 | 15.2 | 3.6 | 124 | 1024 |

**One row per detector — each is the best-performing version of that architecture** (FPR-aware: the
single best trained variant, SAHI/tiled-inference and superseded flat-recipe runs removed). Rows 1,
3, 8, 9, 10 are the DETRs finetuned with their paper optimizer recipe (discriminative backbone LR +
EMA + no-WD-on-norm/bias + grad-clip 0.1 — see the paper-recipe Finding); their earlier flat-recipe
runs have been cleaned up. **Rows 4, 5, 7 are the Pyronear production YOLO11s baselines** — v6
`nimble-narwhal` (the reference baseline all deltas are measured against), plus the two *updated*
releases: `rapid-raccoon-v8.1.0` and the `sensitive-detector-v1.1.0`. Notably **both updated
production models score slightly *below* the v6 baseline on F1** (0.861 and 0.868 vs 0.872), and the
"sensitive" detector — tuned for recall — carries a much higher **FPR 0.258** (2× the baseline).
**LW-DETR-small (row 1) is the top F1 on the board — at less than half rfdetr-nano's params.** ⚠
**`rtdetrv2-r18-paper` (row 10) is a trap:** the paper recipe lifted its F1 to 0.835 (best recall on
the board, 0.893) but its FPR is **0.47** — it fires on ~half of all background frames, operationally
the worst row here, so read it by FPR, not F1. **`deimv2-s` (row 6) is the canonical fair-param
model** — 9.7M params, native 640px, lowest compute/GPU on the board.

The full R&D journey behind these numbers — the DETR checkpoint-selection fix, the flat-vs-paper
recipe story, SAHI/tiled-inference experiments, DEIMv2 resolution/epoch/augmentation ablations — is
in the Findings below and in git history; only the winning version of each model is kept here and in
`data/`. The auto-generated full table (now including the precision-at-recall columns below) is
`data/08_reporting/leaderboard.txt`.

### Precision at fixed recall (recall managed server-side)

Because recall is enforced downstream, the operationally interesting question is **how much precision
each model keeps when forced up to a high recall floor.** These are read off each model's **test**
PR curve (box-matching identical to the main table): for each recall floor, the best box-precision
attainable while keeping recall ≥ that floor (or **—** if the model never reaches that recall on the
test set, even at the lowest cached confidence). Box-precision on GT frames, same as the Precision
column; background false alarms are the separate Image-FPR metric. Sorted by precision @ 0.95 recall.

| Model | P @ R≥0.95 | P @ R≥0.98 | P @ R≥0.99 | max recall |
| :--- | ---: | ---: | ---: | ---: |
| rfdetr-nano | **0.570** | **0.353** | **0.210** | 1.000 |
| yolo11s-sensitive-detector-v1.1.0 | 0.565 | — | — | 0.966 |
| yolo11s-nimble-narwhal-v6.0.0 (baseline) | 0.539 | — | — | 0.956 |
| yolo11s-rapid-raccoon-v8.1.0 | 0.508 | — | — | 0.960 |
| lwdetr-small-paper | 0.484 | 0.152 | 0.080 | 0.990 |
| deimv2-s | 0.468 | 0.147 | 0.053 | 0.993 |
| lwdetr-tiny-paper-full | 0.339 | 0.143 | 0.074 | 0.994 |
| yolo26n-smoke | — | — | — | 0.944 |
| yolo26s-smoke | — | — | — | 0.940 |
| dfine-nano-paper | — | — | — | 0.898 |
| rtdetrv2-r18-paper | — | — | — | 0.893 |
| dfine-small-paper | — | — | — | 0.855 |

**Takeaways:** **`rfdetr-nano` is the clear high-recall winner** — it's the only model that stays
usable at extreme recall (0.35 precision at 0.98, 0.21 at 0.99), roughly 2× every other model in that
regime. At the 0.95 floor the field is closer, with rfdetr-nano and the two production YOLO11s models
(sensitive-detector, nimble-narwhal) leading at ~0.54–0.57. **Five of the twelve — dfine-nano/small,
rtdetrv2-r18, yolo26 s/n — never reach 0.95 recall at all** (their PR curve tops out below it), so
they're unsuitable when a high recall floor is required regardless of their headline F1. Note this
is the opposite ranking from F1: lwdetr-small tops F1 but rfdetr-nano is far better where recall is
pinned high.

### Findings

- **Data + pipeline are sound (no bug).** Our from-scratch YOLO11s control (`yolo11s-smoke-repro`,
  trained on our train split) reproduces the production model almost exactly (F1 0.870 vs 0.872).
  Since it uses ultralytics' own data loading, this clears both the training data and our pipeline.
- **The DETRs' early weakness was a checkpoint-selection bug, not the models or the data.** With
  `eval_loss`-based selection the DETRs looked broken (RT-DETRv2 F1 0.14, D-FINE-nano 0.24,
  D-FINE-small 0.01). Switching to **per-epoch val-F1 selection** (see Approach) fixed every one of
  them: RT-DETRv2 0.14 → **0.74**, D-FINE-nano 0.24 → **0.77**, D-FINE-small 0.01 → **0.55**,
  LW-DETR 0.82 → **0.87**. The architectures and COCO annotations were fine all along — DETR val
  loss simply isn't a usable proxy for detection quality and bottoms out long before F1 peaks.
- **D-FINE-small needed its own optimizer recipe, not just the selection fix.** Even after F1
  selection, D-FINE-small lagged its smaller sibling (0.55 vs nano's 0.77) — suspicious for a bigger
  model. The cause was our *flat* recipe (uniform LR 1e-4, no EMA) applied to a model D-FINE trains
  with a **discriminative LR** (head/encoder/decoder 2e-4, backbone 1e-4) plus **EMA (0.9999)**.
  Retraining with [D-FINE's own hyperparameters](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_s_coco.yml)
  (`dfine-small-paper`) lifted it **0.55 → 0.846** and cut its FPR **0.20 → 0.08** — again confirming
  the data/format were never at fault. Selection and optimizer recipe are *both* load-bearing for
  DETR finetuning. (Opt-in via `--backbone-lr / --ema-decay / --no-wd-on-norm-bias` in `train.py`.)
- **The paper recipe generalizes: it lifted *every* Group-1 DETR, not just D-FINE-small.** The four
  HF DETRs (dfine-nano, lwdetr-tiny, rtdetrv2-r18) had originally shared one *flat* finetuning recipe
  (uniform LR 1e-4, no EMA), which — as with dfine-small — understated them. Retraining each with the
  same paper recipe as `dfine-small-paper` (base LR **2e-4** / backbone **1e-4**, EMA 0.9999, no WD on
  norm/bias, **grad-clip 0.1**) improved all of them: **dfine-nano 0.774 → 0.851** (at just **3.7M
  params**, now within 0.02 of the 9.4M baseline and *lower* FPR), **lwdetr-tiny 0.868 → 0.878** (now
  above baseline). We also added the **LW-DETR *small*** variant (14.2M, similar param class), trained
  with the same recipe: **F1 0.884 — the new top of the board**, edging rfdetr-nano (0.883) at *less
  than half the params* (14.2M vs 31.5M) and beating the baseline on both F1 and FPR. So the earlier
  leaderboard understated the DETR family purely through the flat recipe; three of the four now sit at
  or above the production baseline. Reproduced via `scripts/train_paper_recipe.sh`.
- **Grad-clipping is what makes the high-LR paper recipe safe.** The paper recipe's base LR 2e-4
  diverged to **NaN** on the small dfine-nano backbone when run without tight gradient clipping (the
  trainer had cosine + warmup but no clip). Dropping the LR to 1e-4 instead just *starved* learning
  (nano undertrained to F1 0.54, below even its flat 0.77). The fix was D-FINE's actual pairing —
  **keep LR 2e-4 and clip grad-norm to 0.1** — which trains all four architectures stably. (`train.py`
  now exposes `--max-grad-norm`, default 1.0.)
- **⚠ rtdetrv2-r18-paper: a big F1 gain that's operationally a regression.** The paper recipe raised
  RT-DETRv2-R18 the most on paper (**F1 0.744 → 0.835, +0.092**), but its **image FPR ballooned to
  0.47** — it fires on ~half of all background frames. It bought recall (0.893) with a flood of false
  alarms, so despite the headline F1 it is the *worst* model here for the metric that actually matters
  operationally. A caution that F1 alone can mislead on this task; judge by FPR at equal recall.
- **Full 132-epoch schedule vs early stopping (lwdetr-tiny) — negligible F1, but a cleaner operating
  point.** Rerunning lwdetr-tiny-paper with early stopping disabled (all 132 epochs) moved F1 within
  noise (0.878 → **0.880**) but tightened the model: precision 0.896 → **0.931** and **FPR 0.120 →
  0.091 (−24%)**, trading a little recall. So early stopping wasn't leaving accuracy on the table, but
  the full schedule yields a more precise, fewer-false-alarm model. (`scripts/train_lwdetr_tiny_full.sh`.)
- **DEIMv2-S is the best fair-param detector here — at 640px.** At **9.7M params** (the closest
  match to production's 9.4M) DEIMv2-S reaches **F1 0.864** with a low **0.065** FPR at the **lowest
  compute and GPU memory** on the board (49 GFLOPs, 100 MB), running at its native 640px. Trained
  with its native recipe (Mosaic/mixup aug, EMA, mAP selection) via its own repo.
- **Bigger input is not better for DEIMv2-S — 640 is its sweet spot.** Retrained at 1024 for a
  matched-resolution comparison, it got *slightly worse* (F1 0.864 → **0.853**, FPR 0.065 → 0.086:
  recall edged up but precision fell) while GFLOPs rose **4.4×** (49 → 218) and latency **5.4×**
  (8.4 → 45.5 ms — its ViT-Tiny attention is quadratic in tokens). So unlike the tiling experiment,
  this isn't a resolution-starved model: the small ViT backbone can't turn extra pixels into
  accuracy here, and 640 is both faster and better. Its 640 row is the one to compare.
- **More epochs don't help either — the model was already at its ceiling.** Continuing the 640
  model for +100 epochs (`--tuning` from its best checkpoint) plateaued immediately: val mAP peaked
  at **epoch 12** and never improved, so an early-stop watcher killed it at epoch 27. Test F1 moved
  within noise (0.864 → 0.869) while FPR worsened (0.065 → 0.084) — no real gain. Confirms the
  132-epoch / 640 run had converged. (`scripts/deim_earlystop.py` provides the val-mAP-patience
  early stopping DEIM lacks natively.)
- **A finetuned transformer beats production YOLO on this benchmark.** **RF-DETR-Nano tops the board
  at F1 0.883** and, more importantly for a smoke detector, fires on only **3.9% of background
  frames** vs YOLO's 12.5% — a ~3× lower false-alarm rate at higher recall-adjusted precision. The
  cost is compute: 31.5M params / 329 GFLOPs / 15 ms, the heaviest model here.
- **LW-DETR-tiny is the most attractive accuracy/FP trade.** It matches YOLO on F1 (0.868 vs 0.872)
  at 11.7M params while cutting the false-positive rate to 0.102 (vs 0.125) — and uses the least GPU
  memory on the board (138 MB).
- **D-FINE-nano is the efficiency standout.** F1 0.774 at just **3.7M params / 17.9 GFLOPs** (the
  smallest, lowest-compute model) — ~2.5× fewer params than YOLO. A strong candidate where compute
  budget dominates.
- **At a fair param budget (~9.4–11.7M), the field is tight on F1 but the transformers win on
  false alarms.** Ranked at like-for-like size: YOLO11s control **0.870**, LW-DETR **0.868**,
  DEIMv2-S **0.864**, D-FINE-small-paper **0.846**, YOLO26s **0.834** — all within ~4 F1 points. The
  real separator for a smoke detector is the **false-positive rate**, where the newer detectors
  crush the YOLO11s control (0.131): DEIMv2-S **0.065**, D-FINE-small-paper **0.080**, YOLO26s
  **0.061**, LW-DETR **0.102**. So at equal parameters you don't gain much raw F1 over production
  YOLO, but you can roughly **halve the false-alarm rate** — the metric that matters operationally.
  RF-DETR's higher F1 (0.883) comes at 3.3× the parameters, so it isn't a like-for-like comparison.
- **YOLO26-nano punches above its weight.** At **2.5M params / 15.2 GFLOPs** (the lightest model on
  the board) it reaches F1 0.823 — beating D-FINE-nano (0.774) with fewer params, and the joint-best
  latency/memory profile. The strongest pick under a tight compute budget.
- **Efficiency / latency:** measured back-to-back in one quiet GPU window. YOLO11s is fastest
  (~3 ms). The DETRs cluster at 6–15 ms; note RT-DETRv2's 153 GFLOPs still runs in ~7 ms (it
  parallelizes well), while RF-DETR's 329 GFLOPs at 1024px dominate its 15 ms. Latency is a batch-1
  forward pass (no pre/post-processing or NMS) on an RTX 4090 and is clock-sensitive — treat
  params/GFLOPs as the reliable size/compute metrics.
- **COCO annotation format verified:** the HF DETRs consume standard COCO-detection annotations; a
  GT box round-trips through our converter + the image processor exactly, so the labels fed to every
  DETR are correct.

### Does SAHI (tiled inference) help small-object recall?

[SAHI](https://github.com/obss/sahi) slices each frame into overlapping 640px tiles, runs the
detector on every tile (plus the full frame), and merges the boxes — the standard trick for tiny
objects. We re-ran **every** model with it (`-sahi` rows in `leaderboard.txt`; scripts
`infer_sahi.py`, `infer_sahi_rfdetr.py`, wrapper `sahi_hf.py`), selecting each variant's confidence
on val exactly as for the plain runs.

**Verdict: it does not help this benchmark — 7 of 9 models got worse, and the false-positive rate
rose for 7 of 9.** Test-set F1 (Δ vs plain) and image-FPR:

| Model | plain F1 | SAHI F1 | ΔF1 | plain FPR | SAHI FPR |
|-------|---------:|--------:|----:|----------:|---------:|
| rfdetr-nano | 0.883 | 0.847 | −0.035 | 0.039 | 0.059 |
| yolo11s-nimble-narwhal (prod) | 0.872 | 0.834 | −0.038 | 0.125 | 0.152 |
| yolo11s-smoke-repro | 0.870 | 0.839 | −0.032 | 0.131 | 0.181 |
| lwdetr-tiny | 0.868 | 0.724 | **−0.144** | 0.102 | 0.107 |
| yolo26s-smoke | 0.834 | 0.844 | **+0.011** | 0.061 | 0.104 |
| yolo26n-smoke | 0.823 | 0.825 | +0.002 | 0.106 | 0.167 |
| dfine-nano | 0.774 | 0.691 | −0.082 | 0.152 | 0.183 |
| rtdetrv2-r18 | 0.744 | 0.500 | **−0.244** | 0.170 | 0.140 |
| dfine-small | 0.550 | 0.472 | −0.078 | 0.201 | 0.153 |
| deimv2-s (640 tiles, no resize) | 0.864 | 0.806 | −0.058 | 0.065 | **0.327** |

The last row is a deliberate best-case test: DEIMv2-S is native-640, so tiling a ~1280px frame into
640×640 tiles feeds it at **full resolution with no downscale and no resize** (SAHI keeps tiles
full-size). It *still* lost 0.06 F1 — and the mechanism is unmistakable: **recall barely moved
(0.864→0.846, so native-res tiling did preserve the smoke) while precision collapsed and image-FPR
went 5× (0.065→0.327)**. That pins the SAHI failure on **false-positive multiplication / lost global
context**, not on resolution: even the ideal resolution scenario can't overcome ~6 background tiles
per frame each firing on context-free cloud/haze. Input resolution was never the bottleneck here.

Why SAHI backfires here:
- **The objects aren't small *enough*.** Native frames are ~1280px with a median GT box max-side of
  ~52px (only ~26% are <32px). The detectors already run at 1024px, so smoke is resolved fine
  without tiling; SAHI's usual win (tiny objects in multi-MP images) doesn't apply.
- **Tiling destroys global context.** Smoke is large and diffuse — a 640px tile often contains an
  ambiguous grey smudge the model can only read against the full horizon. The **DETRs suffer most**
  (global attention, fixed-resolution training): RT-DETRv2 −0.24, LW-DETR −0.14. The scale-augmented
  YOLOs are the most robust (yolo26s even nudges +0.01), but not enough to matter.
- **More tiles → more false alarms.** Each background tile is another chance to fire, so image-FPR
  climbed for almost every model — the opposite of what a smoke detector wants.

Net: for this data, **input resolution beats tiling**. If small-object recall needs a further push,
raising train/inference resolution (1280/1536) is the lever to try before tiling; SAHI would only
pay off on genuinely small objects in higher-resolution imagery. (SAHI's per-frame cost is also
~7× a single forward — the `-sahi` rows omit efficiency columns for that reason.)

### Follow-up: does *training* on tiles rescue tiled inference?

The naive SAHI failure above used a **full-frame-trained** model on tiles at inference — it had never
seen context-free background tiles, so it false-fired everywhere (FPR 0.327). Natural fix: **train on
640×640 native tiles too** (so train/infer distributions match and the model learns to stay quiet on
empty tiles), with positive-tile ×2 oversampling and richer augmentation — constrained color jitter
(hue off), gaussian blur, tile-mosaic, smoke context injection (CopyBlend-copy), and **hard-negative
`FPInject`** (pasting pyro-dataset fp/ distractor crops, no box). Full pipeline in
`build_tiled_coco_dataset.py`, `build_fp_crop_pool.py`, `smoke_aug.py`, `deimv2_s_smoke_tiled.yml`;
inference tiled+merged via `infer_sahi_deimv2.py`. DEIMv2-S, 640.

| DEIMv2-S | F1 | Precision | Recall | Image FPR |
|----------|---:|----------:|-------:|----------:|
| plain 640 (full-frame train + infer) | **0.864** | 0.863 | 0.864 | **0.065** |
| naive tiled infer (full-frame model) | 0.806 | 0.769 | 0.846 | 0.327 |
| **tiled train + tiled infer** (`deimv2-s-tiled`) | 0.843 | 0.816 | **0.871** | 0.187 |

**Verdict: it works as intended but still loses to full-frame.** Training on tiles **recovers most of
the naive-tiling loss** (F1 0.806→0.843, FPR 0.327→0.187) and delivers the intended win — **the best
recall of any DEIMv2 variant (0.871)**, i.e. it genuinely catches more small/faint smoke. But
precision/FPR stay far worse than full-frame (0.187 vs 0.065): even trained on them, the ~6
background tiles per frame produce more false alarms than one full-horizon view, because a 640 crop
lacks the global context that disambiguates smoke from haze/cloud. For a smoke detector where the
false-alarm rate is the operational metric, **plain full-frame 640 remains the better choice** — the
recall gain from tiling isn't worth ~3× the FPR. (Consistent with the SAHI and 1024 findings: for
this data, resolution/tiling is not the bottleneck.)

**And the augmentations don't transfer to the full-frame winner either.** Applying the tiled run's
augmentations to the plain full-frame model (`deimv2-s-aug`: fp-injection + copy-mode smoke injection
+ no-hue jitter + gaussian blur, MixUp off — but no tiling) made it **worse**: F1 0.864 → **0.828**,
and — counter to the intent — **FPR rose 0.065 → 0.171** despite adding fp hard-negatives meant to
suppress false alarms. It converged to a lower val-mAP peak (0.354 vs 0.39) at epoch 44. So the extra
augmentation is net-negative here: the **stock DEIMv2-S recipe** (which already includes Mosaic,
MixUp, and CopyBlend *blend*-mode context injection, with hue jitter) is already well-tuned, and the
heavier custom recipe (paste distractors + copy-paste smoke + blur) adds enough distribution shift /
label noise to degrade a model that was already at its ceiling. **`deimv2-s` (stock, 0.864) remains
the best DEIMv2 configuration across every variation tried** (tiling, 1024, +100 epochs, custom augs).

> Note: SAHI's built-in `huggingface` model returns every D-FINE/LW-DETR/RT-DETR box at score 1.0
> (it skips the model's score transform), which would make threshold selection meaningless. The
> `sahi_hf.py` wrapper reuses the model's real `post_process_object_detection`; verified to match the
> non-sliced inference exactly when run unsliced.

> Caveat: the in-training val-F1 selector uses a strided 500-image val subset on a coarse confidence
> grid for speed, so its reported F1 is a noisy *ranking* proxy and reads much lower than the final
> full-val number (e.g. ~0.45 vs 0.85 for the same LW-DETR checkpoint). It ranked checkpoints well
> enough to land results matching production YOLO; a finer in-training eval might recover a slightly
> better epoch.

> RF-DETR-Nano (31.5M) is well above the ~9.4M YOLO baseline — it is the **smallest** RF-DETR
> variant offered, so it is the closest available, not a param-matched comparison. **YOLO26s (9.9M)
> and DEIMv2-S (9.7M) are the fair, like-for-like param matches** to the 9.4M production model.

> DEIMv2-S is scored at its **native 640px**. We also retrained it at 1024 for a matched-resolution
> comparison (`deimv2-s-1024`): it came out **slightly worse** (0.853 vs 0.864) at 4.4× the compute,
> so 640 is not a handicap — it's the model's sweet spot, and the canonical row. Config
> `deimv2_s_smoke_1024.yml` is kept for reproducibility.

> YOLO26 training segfaults (SIGSEGV, exit 139, no traceback) stochastically at a random epoch with
> the ultralytics default of 8 dataloader workers on this CUDA stack (torch 2.12+cu130). `workers: 2`
> makes it rarer but not impossible (it still crashed once at epoch 26), so `train_yolo` runs are
> wrapped in a retry. Worker count does not affect the trained weights, so the comparison stays fair.
> YOLO11s is unaffected.

### DEIMv2-S: initialization & augmentation per run

All DEIMv2-S runs **initialize their backbone from the same DINOv3-distilled ViT-Tiny checkpoint
(`ckpts/vitt_distill.pt`)**; the detector head/encoder/decoder train from scratch on smoke. What
each run starts from and augments with:

| Run | Init | Resolution | Augmentations |
|-----|------|-----------|---------------|
| `deimv2-s` (canonical) | fresh (distilled backbone) | 640 | **stock DEIMv2-S recipe** (below) |
| `deimv2-s-1024` | fresh (distilled backbone) | 1024 | stock recipe |
| `deimv2-s-resume` | **`--tuning` from `deimv2-s` best** (warm-start of the 0.864 weights) | 640 | stock recipe |
| `deimv2-s-tiled` | **fresh (distilled backbone) — NOT from `deimv2-s`** | 640 tiles | tiled recipe (below) |

**`deimv2-s-tiled` is trained from scratch**, not warm-started from the 0.864 model — the only
`--resume` involved was continuing *its own* run after an external interruption (checkpoints saved
every epoch), not loading `deimv2-s` weights. (`deimv2-s-resume` is the one that warm-starts from
`deimv2-s`.)

**Stock DEIMv2-S recipe** (used by `deimv2-s`, `-1024`, `-resume`): Mosaic (2×2, `output_size` 320) ·
`RandomPhotometricDistort` p=0.5 **including hue** · RandomZoomOut · RandomIoUCrop p=0.8 ·
HorizontalFlip · multi-scale training (480–800) · **MixUp** p=0.5 (ep 4–64) · **CopyBlend blend-mode**
p=0.5 (ep 4–120) — i.e. it already does GT-crop context injection · policy stages [4,64,120]
(no-aug warmup → mosaic → no-mosaic → no-aug tail). EMA 0.9999.

**Tiled recipe** (`deimv2-s-tiled`, `deimv2_s_smoke_tiled.yml`): trained on 640×640 native tiles
(`build_tiled_coco_dataset.py`, positives ×2-oversampled). Changes vs stock: color jitter with **hue
disabled** and small ranges; **RandomGaussianBlur** p=0.3 (new); **CopyBlend switched to `copy`-mode +
`with_expand`** (harder context injection); **FPInject** (new — pastes hard-negative false-positive
crops from `build_fp_crop_pool.py`, adds no box); **MixUp disabled**. Mosaic/zoom/crop/flip/multi-
scale retained. Inference is tiled + merged (`infer_sahi_deimv2.py`).

## How to Reproduce

```bash
cd experiments/detectors/detector-leaderboard
make install

# One-time: import the flat YOLO splits (images + labels). test comes from
# yolo_test; train/val from yolo_train_val. Needs pyronear dataset S3 creds.
REPO=https://github.com/pyronear/pyro-dataset
REV=v4.0.0-corrected
for mod in images labels; do
  AWS_PROFILE=pyro uv run dvc import $REPO data/processed/yolo_test/$mod/test \
    -o data/01_raw/datasets/test/$mod --rev $REV
  AWS_PROFILE=pyro uv run dvc import $REPO data/processed/yolo_train_val/$mod/train \
    -o data/01_raw/datasets/train/$mod --rev $REV
  AWS_PROFILE=pyro uv run dvc import $REPO data/processed/yolo_train_val/$mod/val \
    -o data/01_raw/datasets/val/$mod --rev $REV
done
# Collaborators with the .dvc files committed can instead just pull:
AWS_PROFILE=pyro uv run dvc pull

uv run dvc repro            # trains the DETR + YOLO-control models (GPU, multi-hour)
uv run dvc metrics show
cat data/08_reporting/leaderboard.txt
```

RF-DETR runs outside DVC, in its isolated venv (after the splits are imported):

```bash
uv venv .rfdetr-venv && uv pip install --python .rfdetr-venv/bin/python "rfdetr[train,loggers]"
uv run python scripts/build_rfdetr_dataset.py \
  --train-dir data/01_raw/datasets/train --val-dir data/01_raw/datasets/val \
  --test-dir data/01_raw/datasets/test --output-dir data/05_model_input/rfdetr_coco
PY=.rfdetr-venv/bin/python
$PY scripts/train_rfdetr.py --dataset-dir data/05_model_input/rfdetr_coco \
  --output-dir data/06_models/rfdetr-nano --resolution 1024 --epochs 100 \
  --batch-size 2 --grad-accum-steps 8 --early-stop-patience 20
for split in val test; do
  $PY scripts/infer_rfdetr.py --data-dir data/01_raw/datasets/$split \
    --checkpoint data/06_models/rfdetr-nano/checkpoint_best_total.pth \
    --output-file data/02_intermediate/rfdetr-nano/${split}_predictions.json --resolution 1024
done
$PY scripts/profile_rfdetr.py --model-name rfdetr-nano \
  --checkpoint data/06_models/rfdetr-nano/checkpoint_best_total.pth \
  --output-file data/07_model_output/rfdetr-nano/profile.json --resolution 1024
# evaluate (main venv, backend-agnostic) then rebuild the leaderboard:
uv run python scripts/evaluate.py --model-name rfdetr-nano \
  --val-predictions data/02_intermediate/rfdetr-nano/val_predictions.json --val-dir data/01_raw/datasets/val \
  --test-predictions data/02_intermediate/rfdetr-nano/test_predictions.json --test-dir data/01_raw/datasets/test \
  --output-dir data/07_model_output/rfdetr-nano --iou-threshold 0.1
uv run python scripts/leaderboard.py --results-dir data/07_model_output --output-dir data/08_reporting
```

D-FINE-small with the paper optimizer recipe (`dfine-small-paper`), with W&B logging:

```bash
uv run python scripts/train.py --checkpoint ustc-community/dfine-small-coco \
  --train-dir data/01_raw/datasets/train --val-dir data/01_raw/datasets/val \
  --output-dir data/06_models/dfine-small-paper \
  --image-size 1024 --epochs 132 --batch-size 8 \
  --learning-rate 2e-4 --backbone-lr 1e-4 --weight-decay 1e-4 --no-wd-on-norm-bias \
  --ema-decay 0.9999 --early-stop-patience 20 --seed 42 \
  --wandb-project detector-leaderboard --wandb-run-name dfine-small-paper --log-images
# then infer (backend dfine) -> evaluate -> profile -> leaderboard, as for the DVC dfine rows.
```

DEIMv2-S runs outside DVC in its own venv on the cloned repo (reuses `rfdetr_coco`):

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2 deimv2_repo
uv venv .deimv2-venv --python 3.11
uv pip install --python .deimv2-venv/bin/python -r deimv2_repo/requirements.txt gdown
# ViT-Tiny distilled backbone -> ckpts/ (config deimv2_s_smoke.yml expects it there):
.deimv2-venv/bin/gdown 1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs -O deimv2_repo/ckpts/vitt_distill.pt
# config configs/deimv2/deimv2_s_smoke.yml: num_classes=2, remap off, rfdetr_coco paths.
cd deimv2_repo && ../.deimv2-venv/bin/torchrun --nproc_per_node=1 train.py \
  -c configs/deimv2/deimv2_s_smoke.yml --use-amp --seed=0 && cd ..
# infer (val+test) + profile in the venv, then evaluate + leaderboard in the main env:
.deimv2-venv/bin/python scripts/infer_deimv2.py --config deimv2_repo/configs/deimv2/deimv2_s_smoke.yml \
  --checkpoint deimv2_repo/outputs/deimv2_s_smoke/best_stg2.pth \
  --data-dir data/01_raw/datasets/test \
  --output-file data/02_intermediate/deimv2-s/test_predictions.json  # + val
.deimv2-venv/bin/python scripts/profile_deimv2.py --config <cfg> --checkpoint <ckpt> \
  --model-name deimv2-s --output-file data/07_model_output/deimv2-s/profile.json
uv run python scripts/evaluate.py --model-name deimv2-s ...  # then leaderboard.py
# mirror DEIMv2's TensorBoard logs into W&B:
uv run python scripts/sync_tb_to_wandb.py --logdir deimv2_repo/outputs/deimv2_s_smoke/summary \
  --project detector-leaderboard --run-name deimv2-s
```

Optional — sliced (SAHI) inference for any model (see the SAHI finding above). Emits `<model>-sahi`
rows scored by the same evaluator:

```bash
uv pip install sahi   # main venv; and: uv pip install --python .rfdetr-venv/bin/python sahi
# YOLO (ultralytics) or HF DETR (hf_detr):
uv run python scripts/infer_sahi.py --backend ultralytics \
  --model-path data/06_models/yolo26s-smoke/best.pt \
  --data-dir data/01_raw/datasets/test \
  --output-file data/02_intermediate/yolo26s-smoke-sahi/test_predictions.json \
  --slice-size 640 --overlap 0.2 --confidence-threshold 0.01
# RF-DETR (isolated venv): scripts/infer_sahi_rfdetr.py, same flags + --checkpoint/--resolution.
# Then evaluate (val+test) with --model-name <model>-sahi and rebuild the leaderboard as above.
```

DEIMv2-S tiled training + tiled inference (`deimv2-s-tiled`; apply `deimv2_patch/` first — see its
README):

```bash
# 1. Build the 640-tile COCO dataset (positives x2-oversampled) + fp distractor pool:
uv run python scripts/build_tiled_coco_dataset.py \
  --train-dir data/01_raw/datasets/train --val-dir data/01_raw/datasets/val \
  --output-dir data/05_model_input/tiled_coco --tile 640 --overlap 0.2 --pos-oversample 2
AWS_PROFILE=pyro uv run dvc import --rev v4.0.0-corrected https://github.com/pyronear/pyro-dataset \
  data/processed/fp_yolo/images/train -o data/01_raw/fp_frames/train   # + val
uv run python scripts/build_fp_crop_pool.py --fp-dir data/01_raw/fp_frames \
  --output-dir data/05_model_input/fp_crops
# 2. Train (fresh from distilled backbone; early-stop watcher); config = configs/deimv2/deimv2_s_smoke_tiled.yml
cd deimv2_repo && ../.deimv2-venv/bin/torchrun --nproc_per_node=1 train.py \
  -c configs/deimv2/deimv2_s_smoke_tiled.yml --use-amp --seed=0 & cd ..
uv run python scripts/deim_earlystop.py --log deimv2_repo/outputs/deimv2_s_smoke_tiled/log.txt \
  --pid <torchrun_pid> --match "train.py -c configs/deimv2/deimv2_s_smoke_tiled" --patience 5 --min-epochs 5
# 3. Tiled inference + merge -> evaluate -> leaderboard (as model-name deimv2-s-tiled):
.deimv2-venv/bin/python scripts/infer_sahi_deimv2.py --config deimv2_repo/configs/deimv2/deimv2_s_smoke_tiled.yml \
  --checkpoint deimv2_repo/outputs/deimv2_s_smoke_tiled/best_stg1.pth \
  --data-dir data/01_raw/datasets/test --output-file data/02_intermediate/deimv2-s-tiled/test_predictions.json  # + val
uv run python scripts/evaluate.py --model-name deimv2-s-tiled ...  # then leaderboard.py
```

## Adding a Detector

- **YOLO (ultralytics):** add an entry under `ultralytics_detectors:` in `params.yaml`
  (`model_repo` + `model_filename` on HuggingFace Hub). For a YOLO trained here, add under
  `yolo_trained_detectors:` (an `arch` init `.pt`).
- **HF DETR (D-FINE / LW-DETR / RT-DETR / …):** add an entry under `dfine_detectors:` or
  `hf_detr_detectors:` (a COCO-pretrained `checkpoint`); it is finetuned with val-F1 selection by
  the `train` / `train_hf` stage. Any `AutoModelForObjectDetection`-compatible checkpoint works.
- **RF-DETR / DEIMv2:** run via their isolated-venv scripts above (not DVC); they emit the same
  `predictions.json` / `profile.json`, so the evaluator and leaderboard treat them like any other row.

For the DVC-backed registries, the key becomes the output dir name and the leaderboard `model_name`;
run `uv run dvc repro` — DVC's `foreach` picks up the new detector automatically. The leaderboard
auto-discovers any `data/07_model_output/<model>/metrics.json` (+ optional `profile.json`).

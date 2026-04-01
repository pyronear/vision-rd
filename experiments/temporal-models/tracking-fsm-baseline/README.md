# 🔥 Tracking FSM Baseline

## 🎯 Objective

Reduce false positives from the production YOLO smoke detector by requiring temporal persistence: a detection must appear in multiple consecutive frames before raising an alarm. Inspired by the [FLAME paper](../../literature_survey/notes/2024-flame.md) (Gragnaniello 2024), which achieves +18% precision using a tracking-based motion filter.

## 🧠 Approach

1. Run production YOLO11s on every frame of each sequence
2. Pad short sequences by symmetrically repeating boundary frames (min 10 frames)
3. Filter detections by confidence threshold and max bounding box area
4. Match detections across consecutive frames using greedy IoU matching
5. Confirm a detection only if it persists for `min_consecutive` frames
6. Optionally apply post-confirmation filters (mean confidence, area change)
7. Classify a sequence as positive if any confirmed track survives all filters

No ML training needed — purely rule-based temporal filtering.

### Post-confirmation rules (toggleable)

Each rule has a boolean flag and a threshold. Disabled by default.

- **Gap tolerance** (`max_misses`): Allow missed frames before dropping a track. Targets FN recovery.
- **Confidence filter** (`use_confidence_filter` + `min_mean_confidence`): Reject confirmed tracks with low average detection confidence. Targets FP reduction.
- **Area change filter** (`use_area_change_filter` + `min_area_change`): Reject confirmed tracks where the detection area didn't grow (smoke expands, static FP doesn't). Targets FP reduction.

## 📊 Data

Imported from [pyro-dataset](https://github.com/pyronear/pyro-dataset) v2.2.0
via `dvc import`, then truncated to max 20 frames per sequence:

- **Train**: ~1,034 WF + ~1,433 FP sequences (Pyronear only; 2,865 total incl. external)
- **Val**: ~112 WF + ~147 FP sequences (Pyronear only; 294 total incl. external)
- Layout: `data/01_raw/datasets/{train,val}/{wildfire,fp}/sequence_name/{images,labels}/`
- Ground truth: sequence-level binary, inferred from label format
  - WF (positive): 5-column labels (`class cx cy w h`) — human annotations
  - FP (negative): 6-column labels (`class cx cy w h confidence`) — prior YOLO predictions on non-smoke scenes

## 📏 Evaluation

All metrics are computed **at the sequence level**, not per-frame, on both all sequences and Pyronear-only subsets.

The tracker processes all frames in a sequence and produces a single binary prediction:
- **Positive** if any tracked detection is confirmed (persisted for `min_consecutive` consecutive frames)
- **Negative** otherwise

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **WF sequence** (real smoke) | TP | FN |
| **FP sequence** (false alarm) | FP | TN |

**Metrics:**
- **Precision** — of sequences where we raised an alarm, how many had real smoke?
- **Recall** — of sequences with real smoke, how many did we catch?
- **F1** — harmonic mean of precision and recall
- **FPR** — of FP sequences, how many falsely raised an alarm?
- **Time-to-detection (TTD)** — seconds from first frame to tracker confirmation (for true positives)

**YOLO-only baseline:** any YOLO detection in any frame counts as a positive prediction for that sequence. This gives an upper bound on recall and a lower bound on precision.

## 📈 Results

Best parameters from sweep on train/pyronear (`conf=0.3, iou=0.1, min_consecutive=5, max_detection_area=0.05`):

### Pyronear sequences (what matters for production)

**Validation (259 sequences: 112 WF + 147 FP):**

| Method | Precision | Recall | F1 | FPR | Mean TTD |
|---|---|---|---|---|---|
| YOLO-only | 0.775 | 0.982 | 0.866 | 0.218 | — |
| Tracking | 0.893 | 0.964 | 0.927 | 0.088 | 56s |

**Training (2,467 sequences: 1,034 WF + 1,433 FP):**

| Method | Precision | Recall | F1 | FPR | Mean TTD |
|---|---|---|---|---|---|
| YOLO-only | 0.696 | 0.987 | 0.816 | 0.311 | — |
| Tracking | 0.873 | 0.962 | 0.915 | 0.101 | 126s |

### ✅ Key improvements over YOLO-only (val/pyronear)

- **+11.8pp precision** (0.775 → 0.893)
- **-13.0pp FPR** (0.218 → 0.088) — 60% reduction in false alarms
- **-1.8pp recall** (0.982 → 0.964) — minimal cost
- **Mean TTD: 56s** (median: 30s)

## ⚙️ Pipeline

```
truncate (01_raw/datasets_full → 01_raw/datasets)
  → infer (01_raw → 02_intermediate)
    → pad (02_intermediate → 03_primary)
      → track (03_primary → 07_model_output)
        → evaluate / sweep / ablation (→ 08_reporting)
```

Each evaluate, sweep, and ablation stage runs on both all sequences and Pyronear-only subsets.

## 🚀 How to Reproduce

```bash
cd experiments/temporal-models/tracking-fsm-baseline
make install

# Dataset is imported via DVC from pyro-dataset v2.2.0:
#   uv run dvc import https://github.com/pyronear/pyro-dataset \
#       data/processed/sequential_train_val/train \
#       -o data/01_raw/datasets_full/train --rev v2.2.0
#   uv run dvc import https://github.com/pyronear/pyro-dataset \
#       data/processed/sequential_train_val/val \
#       -o data/01_raw/datasets_full/val --rev v2.2.0
# The .dvc files are committed — just pull:
uv run dvc pull

uv run dvc repro
uv run dvc metrics show
```

### Parameter sweep

Sweeps are DVC stages with multiprocessing. They explore a grid of base params + rule params (28,800 combinations).

```bash
# Run sweep stages only
uv run dvc repro sweep_train_pyronear sweep_val_pyronear
```

### Ablation study

Tests all 8 on/off combinations of the 3 rules with fixed base params from `params.yaml`:

```bash
uv run dvc repro ablation_val_pyronear
```

### 🔍 Visualize results with FiftyOne

Browse all predictions interactively — filter by TP/FP/FN/TN, inspect YOLO detections frame-by-frame, and group by sequence.

```bash
make fiftyone
```

This builds FiftyOne datasets for both train and val splits, then launches the web UI at `http://localhost:5151`. Each frame is a sample with three detection overlay fields, color-coded:

- **Green** — ground truth boxes (human annotations)
- **Purple** — prior YOLO predictions (from label files)
- **Red** — current model YOLO detections

Use the **tags sidebar** to filter by category (e.g. click `false_positive` to see only FP sequences), and the **confidence slider** on `yolo_detections` to interactively threshold. Switch between `tracking-fsm-val` and `tracking-fsm-train` datasets via the dropdown.

### 🎛️ Tunable parameters

| Parameter | Description | Default | Sweep range |
|---|---|---|---|
| `pad.min_sequence_length` | Min frames per sequence (pad if shorter) | 10 | — |
| `track.confidence_threshold` | Min YOLO confidence to keep a detection | 0.3 | 0.1–0.5 |
| `track.iou_threshold` | Min IoU to match detections across frames | 0.1 | 0.1–0.5 |
| `track.min_consecutive` | Consecutive frames needed to confirm | 5 | 1–5 |
| `track.max_detection_area` | Max normalized bbox area (w*h) | 0.05 | None, 0.05–0.15 |
| `track.max_misses` | Frames a track can survive without match | 0 | 0–2 |
| `track.use_confidence_filter` | Enable mean-confidence post-filter | false | — |
| `track.min_mean_confidence` | Min mean confidence for confirmed tracks | 0.3 | 0.0–0.5 |
| `track.use_area_change_filter` | Enable area-growth post-filter | false | — |
| `track.min_area_change` | Min last/first area ratio | 1.1 | 0.0–1.2 |

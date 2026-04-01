# MTB Change Detection

Temporal model for smoke false-positive reduction using MTB (Moving object pixels To Bounding box pixels) change detection, inspired by the [SlowFastMTB paper](../../literature_survey/notes/2024-slowfast-mtb.md) (Choi, Kim & Oh, 2022).

## Approach

Combines YOLO smoke detection with pixel-wise frame differencing to reject static false positives:

1. Run production YOLO11s on every frame of each sequence
2. Pad short sequences by symmetrically repeating boundary frames (min 10 frames)
3. Filter detections by confidence threshold and max bounding box area
4. Compute pixel-wise change between consecutive frames (MTB)
5. Validate detections: only those with sufficient change in their bounding box region are kept
6. Match detections across consecutive frames using greedy IoU matching
7. Confirm a detection only if it persists for `min_consecutive` change-validated frames
8. Classify a sequence as positive if any confirmed track survives

The change validation step is the key difference from the [tracking FSM baseline](../tracking-fsm-baseline/): instead of relying solely on temporal persistence, detections must show real pixel-level change in their bounding box region.

## Dataset

Imported from [pyro-dataset](https://github.com/pyronear/pyro-dataset) v2.2.0
via `dvc import`, then truncated to max 20 frames per sequence:

- **Train**: ~1,070 WF + ~1,466 FP sequences (Pyronear only; 2,933 total incl. external)
- **Val**: ~116 WF + ~151 FP sequences (Pyronear only; 302 total incl. external)
- Layout: `data/01_raw/datasets/{train,val}/{wildfire,fp}/sequence_name/{images,labels}/`
- Ground truth: sequence-level binary, inferred from label format
  - WF (positive): 5-column labels (`class cx cy w h`) — human annotations
  - FP (negative): 6-column labels (`class cx cy w h confidence`) — prior YOLO predictions on non-smoke scenes

## Evaluation

All metrics are computed **at the sequence level**, not per-frame, on both all sequences and Pyronear-only subsets.

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

**YOLO-only baseline:** any YOLO detection in any frame counts as a positive prediction. Upper bound on recall, lower bound on precision.

## Results

Best parameters from sweep on train/pyronear (`pixel_threshold=10, min_change_ratio=0.01, conf=0.01, iou=0.1, min_consecutive=2`):

### Pyronear sequences

**Validation (267 sequences: 116 WF + 151 FP):**

| Method | Precision | Recall | F1 | FPR | Mean TTD |
|---|---|---|---|---|---|
| YOLO-only | 0.578 | 0.991 | 0.730 | 0.556 | — |
| MTB tracking | 0.682 | 0.922 | 0.784 | 0.331 | 58s |

**Training (2,536 sequences: 1,070 WF + 1,466 FP):**

| Method | Precision | Recall | F1 | FPR | Mean TTD |
|---|---|---|---|---|---|
| YOLO-only | 0.525 | 0.971 | 0.682 | 0.641 | — |
| MTB tracking | 0.617 | 0.930 | 0.742 | 0.421 | 87s |

### Key improvements over YOLO-only (val/pyronear)

- **+10.4pp precision** (0.578 → 0.682)
- **-22.5pp FPR** (0.556 → 0.331) — 40% reduction in false alarms
- **-6.9pp recall** (0.991 → 0.922) — moderate cost
- **Mean TTD: 58s** (median: 15s)

## Pipeline

```
truncate (01_raw/datasets_full → 01_raw/datasets)
  → infer (01_raw → 02_intermediate)
    → pad (02_intermediate → 03_primary)
      → track (03_primary → 07_model_output)
        → evaluate / sweep (→ 08_reporting)
      → package (→ 06_models/model.zip)
```

Each evaluate and sweep stage runs on both all sequences and Pyronear-only subsets.

## How to Reproduce

```bash
cd experiments/temporal-models/mtb-change-detection
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

Sweeps are DVC stages with multiprocessing. They explore a grid of MTB + tracker params.

```bash
# Run sweep stages only
uv run dvc repro sweep_train_pyronear sweep_val_pyronear
```

### Packaging

Bundle YOLO weights and all config into a single deployable `.zip` archive:

```bash
uv run dvc repro package
```

### Visualize results with FiftyOne

Browse all predictions interactively — filter by TP/FP/FN/TN, inspect YOLO detections frame-by-frame, and group by sequence.

```bash
make fiftyone
```

This builds FiftyOne datasets for both train and val splits, then launches the web UI at `http://localhost:5151`. Each frame is a sample with three detection overlay fields, color-coded:

- **Green** — ground truth boxes (human annotations)
- **Purple** — prior YOLO predictions (from label files)
- **Red** — current model YOLO detections

Use the **tags sidebar** to filter by category (e.g. click `false_positive` to see only FP sequences), and the **confidence slider** on `yolo_detections` to interactively threshold.

### Notebooks

```bash
make notebook
```

- `01-visualize-change-masks.ipynb` — bird's-eye view of frame-to-frame change masks across sequences (raw frames + binary change masks side by side)

## Tunable parameters

| Parameter | Description | Default | Sweep range |
|---|---|---|---|
| `change.pixel_threshold` | Per-pixel intensity diff threshold for change mask | 10 | 3–40 |
| `change.min_change_ratio` | Min fraction of changed pixels in bbox to validate | 0.01 | 0.001–0.1 |
| `pad.min_sequence_length` | Min frames per sequence (pad if shorter) | 10 | — |
| `track.confidence_threshold` | Min YOLO confidence to keep a detection | 0.01 | 0.01–0.3 |
| `track.iou_threshold` | Min IoU to match detections across frames | 0.1 | 0.1–0.3 |
| `track.min_consecutive` | Consecutive frames needed to confirm | 2 | 1–5 |
| `track.max_detection_area` | Max normalized bbox area (w*h) | 1.0 | None, 0.05, 0.10 |
| `track.max_misses` | Frames a track can survive without match | 0 | 0–1 |

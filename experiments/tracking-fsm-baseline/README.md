# Tracking FSM Baseline

## Objective

Reduce false positives from the production YOLO smoke detector by requiring temporal persistence: a detection must appear in multiple consecutive frames before raising an alarm. Inspired by the [FLAME paper](../../literature_survey/notes/2024-flame.md) (Gragnaniello 2024), which achieves +18% precision using a tracking-based motion filter.

## Approach

**Level 0 — Simple Persistence Filter** (current):
1. Run production YOLO11s on every frame of each sequence
2. Match detections across consecutive frames using greedy IoU matching
3. Confirm a detection only if it persists for `min_consecutive` frames
4. Classify a sequence as positive if any detection is confirmed

No ML training needed — purely rule-based temporal filtering.

## Data

Sequential dataset from `pyro-dataset`, truncated to max 20 frames per sequence:
- Train: ~1,433 WF + ~1,432 FP sequences (50/50 balanced)
- Val: ~147 WF + ~147 FP sequences
- Ground truth: sequence-level binary, inferred from label format
  - WF (positive): 5-column labels (`class cx cy w h`) — human annotations
  - FP (negative): 6-column labels (`class cx cy w h confidence`) — prior YOLO predictions on non-smoke scenes

## Evaluation

All metrics are computed **at the sequence level**, not per-frame.

The tracker processes all frames in a sequence and produces a single binary prediction:
- **Positive** if any tracked detection is confirmed (persisted for `min_consecutive` consecutive frames)
- **Negative** otherwise

A sequence is then classified as:

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

## Results

Best parameters from sweep on train, evaluated on val (`conf=0.4, iou=0.1, min_consecutive=2`):

| Method | Precision | Recall | F1 | FPR | Mean TTD |
|---|---|---|---|---|---|
| YOLO-only | 0.785 | 0.993 | 0.877 | 0.272 | — |
| Tracking | 0.876 | 0.912 | 0.893 | 0.129 | 78s |

## How to Reproduce

```bash
cd experiments/tracking-fsm-baseline
make install
uv run dvc repro
uv run dvc metrics show
```

### Parameter sweep

```bash
# Sweep on train set (fast — reuses cached inference)
uv run python scripts/sweep.py \
    --infer-dir data/02_intermediate/train \
    --data-dir data/01_raw/datasets/train \
    --output-dir data/08_reporting/sweep

# Sweep on val set
uv run python scripts/sweep.py \
    --infer-dir data/02_intermediate/val \
    --data-dir data/01_raw/datasets/val \
    --output-dir data/08_reporting/sweep_val
```

### Tunable parameters

| Parameter | Description | Default | Sweep range |
|---|---|---|---|
| `track.confidence_threshold` | Min YOLO confidence to keep a detection | 0.25 | 0.1–0.5 |
| `track.iou_threshold` | Min IoU to match detections across frames | 0.3 | 0.1–0.5 |
| `track.min_consecutive` | Consecutive frames needed to confirm | 2 | 1–5 |

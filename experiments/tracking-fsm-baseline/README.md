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

Sequential dataset from `pyro-datasets`, truncated to max 20 frames per sequence:
- Train: ~1,433 WF + ~1,432 FP sequences (50/50 balanced)
- Val: ~147 WF + ~147 FP sequences
- Ground truth: sequence-level binary (WF registry lookup)

## Results

_Run `uv run dvc repro` and check `data/08_reporting/val/metrics.json`._

## How to Reproduce

```bash
cd experiments/tracking-fsm-baseline
make install
uv run dvc repro
uv run dvc metrics show
```

### Parameter sweep

```bash
uv run dvc exp run -S track.min_consecutive=3
uv run dvc exp run -S track.iou_threshold=0.2
uv run dvc exp show
```

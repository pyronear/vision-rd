# MTB Change Detection

Temporal model for smoke false-positive reduction using MTB (Moving object pixels To Bounding box pixels) change detection, inspired by the SlowFastMTB paper (Choi, Kim & Oh, 2022).

## Approach

Combines YOLO smoke detection with pixel-wise frame differencing to reject static false positives:

1. YOLO detects smoke candidates in each frame
2. MTB computes pixel-wise change between consecutive frames
3. Detections are validated: only those with significant change in their bounding box region are kept
4. An IoU-based tracker confirms persistent change-validated detections across N consecutive frames

## Dataset

Sequential dataset from `pyro-dataset`, truncated to max 20 frames per sequence. Organized as `{train,val}/{wildfire,fp}/<sequence>/`.

## Quick Start

```bash
make install
make test
```

## Pipeline

```bash
uv run dvc repro
uv run dvc metrics show
```

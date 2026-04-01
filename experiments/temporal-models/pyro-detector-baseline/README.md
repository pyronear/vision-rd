# Pyro-Detector Baseline

## Objective

Evaluate the production pyro-engine detection pipeline (`pyro-predictor`) as a
`TemporalModel` baseline. Provides a reference point for comparing new temporal
models against the actual production system.

## Approach

1. Wrap `pyro-predictor`'s `Predictor` class (YOLO ONNX detection + per-camera
   sliding-window temporal smoothing) as a pyrocore `TemporalModel` subclass.
2. Run the Predictor on every frame of each sequence, using a unique camera ID
   per sequence to isolate sliding-window state.
3. Classify a sequence as positive when the Predictor's aggregated confidence
   exceeds `conf_thresh`.
4. Compute sequence-level metrics (precision, recall, F1, FPR, TTD).

The Predictor mirrors production behavior (defaults from `pyro-engine`'s
`Engine` class):
- YOLO ONNX model for single-frame detection (same weights as other experiments,
  from `pyronear/yolo11s_mighty-mongoose_v5.1.0`)
- Per-camera sliding window of 7 consecutive frames
- Aggregated confidence output in [0, 1]

## Data

Imported from [pyro-dataset](https://github.com/pyronear/pyro-dataset) v2.2.0
via `dvc import` (sequential_train_val split):

- **Train**: 1,467 wildfire + 1,467 FP sequences
- **Val**: 151 wildfire + 151 FP sequences
- Layout: `data/01_raw/datasets/{train,val}/{wildfire,fp}/sequence_name/{images,labels}/`
- Ground truth: inferred from parent directory name (`wildfire/` = positive, `fp/` = negative)

## Results

_To be filled after running the pipeline._

## How to Reproduce

```bash
cd experiments/temporal-models/pyro-detector-baseline
make install

# Dataset is imported via DVC from pyro-dataset v2.2.0:
#   uv run dvc import https://github.com/pyronear/pyro-dataset \
#       data/processed/sequential_train_val/train \
#       -o data/01_raw/datasets/train --rev v2.2.0
#   uv run dvc import https://github.com/pyronear/pyro-dataset \
#       data/processed/sequential_train_val/val \
#       -o data/01_raw/datasets/val --rev v2.2.0
# The .dvc files are committed — just pull:
uv run dvc pull

uv run dvc repro
uv run dvc metrics show
```

## Pipeline

```
prepare (download ONNX model from HuggingFace)
  -> predict (01_raw -> 07_model_output)
    -> evaluate (-> 08_reporting)
```

## Parameters

All defaults match the production `Engine` class in
[pyro-engine](https://github.com/pyronear/pyro-engine).

| Parameter | Default | Description |
|---|---|---|
| `predict.conf_thresh` | 0.35 | **Alert threshold.** The Predictor accumulates detections across the sliding window and computes an aggregated confidence score. An alert fires when this score exceeds `conf_thresh`. Also used to filter individual detections before aggregation. |
| `predict.model_conf_thresh` | 0.05 | **YOLO detection threshold.** Minimum confidence for the YOLO model to report a bounding box. Kept low so the temporal layer sees all candidate detections. |
| `predict.nb_consecutive_frames` | 7 | **Sliding window size.** Number of recent frames the Predictor keeps in its per-camera state for temporal smoothing. A detection must persist across enough of these frames to trigger an alert. |
| `predict.max_bbox_size` | 0.4 | **Max detection width.** Detections wider than this fraction of the image are discarded (filters out large false positives like clouds). |

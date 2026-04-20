# Pyro-Detector Baseline

## Objective

Evaluate the production pyro-engine detection pipeline (`pyro-predictor`) as a
`TemporalModel` baseline. Provides a reference point for comparing new temporal
models against the actual production system.

## Approach

1. Wrap `pyro-predictor`'s `Predictor` class (YOLO ONNX detection + per-camera
   sliding-window temporal smoothing) as a pyrocore `TemporalModel` subclass.
2. Run YOLO inference once per frame and cache detections (`infer` stage).
3. Replay cached detections through the Predictor's temporal logic (`predict`
   stage) — classifies a sequence as positive when the aggregated confidence
   exceeds `conf_thresh`.
4. Grid-search `conf_thresh` x `nb_consecutive_frames` on cached detections
   (`sweep` stage) to find optimal parameters.
5. Compute sequence-level metrics (precision, recall, F1, FPR, TTD).

The Predictor mirrors production behavior:
- YOLO ONNX model for single-frame detection (weights from
  `pyronear/yolo11s_nimble-narwhal_v6.0.0`)
- Per-camera sliding window for temporal smoothing
- Aggregated confidence output in [0, 1]

## Data

Imported from [pyro-dataset](https://github.com/pyronear/pyro-dataset) v2.2.0
via `dvc import` (sequential_train_val split):

- **Train**: 1,467 wildfire + 1,467 FP sequences
- **Val**: 151 wildfire + 151 FP sequences
- Layout: `data/01_raw/datasets/{train,val}/{wildfire,fp}/sequence_name/{images,labels}/`
- Ground truth: inferred from parent directory name (`wildfire/` = positive, `fp/` = negative)

## Results

### Evaluate (optimized params: conf=0.2, nb_frames=4)

| Split | Method | Precision | Recall | F1 | FPR | Median TTD (frames) |
|---|---|---|---|---|---|---|
| val/all | Single-frame | 0.853 | 0.960 | 0.903 | 0.166 | -- |
| val/all | Predictor | 0.892 | 0.934 | **0.913** | 0.113 | 1 |
| train/all | Single-frame | 0.820 | 0.981 | 0.893 | 0.215 | -- |
| train/all | Predictor | 0.849 | 0.971 | **0.906** | 0.172 | 1 |

#### Note on `nb_consecutive_frames` semantics

Despite the name, **`nb_consecutive_frames` is the sliding-window size, not a hard "wait N frames before alarm" gate** — see `pyro_predictor/predictor.py::_update_states`. The aggregation is roughly:

- `conf_th = conf_thresh * nb_consecutive_frames` (threshold on the NMS-summed score)
- `strong_detection = sum(iou > 0, axis=0) >= nb_consecutive_frames // 2` (= 2 when `nb_frames=4`)
- `conf = max(summed_scores) / (nb_consecutive_frames + 1)`; alarm when `conf > conf_thresh`

The `strong_detection` check counts **overlapping boxes across the window**, including multiple co-located boxes returned on the current frame. YOLO often emits 2+ overlapping high-confidence boxes for a clear smoke plume, so the ≥ 2 check can pass on **frame 0 alone**. If the sum of overlapping scores exceeds `conf_th`, the alarm fires immediately → `trigger_frame_index = 0`.

**Consequence**: median TTD = 1 frame on the leaderboard test set is a real artifact of the production algorithm, not of the TTD migration. If you actually need "wait N consecutive frames before alarm" semantics for the baseline, that's an algorithm change in `pyro-engine`, not a leaderboard/evaluator change.

### Sweep top configs (val/all, ranked by F1)

| conf | nb_frames | Precision | Recall | F1 | FPR | Mean TTD (frames) |
|---|---|---|---|---|---|---|
| 0.20 | 4 | 0.892 | 0.934 | **0.913** | 0.113 | 2.0 |
| 0.25 | 3 | 0.897 | 0.927 | 0.912 | 0.106 | 1.8 |
| 0.20 | 5 | 0.892 | 0.927 | 0.909 | 0.113 | 2.2 |
| 0.30 | 2 | 0.892 | 0.927 | 0.909 | 0.113 | 1.7 |

Production defaults (conf=0.35, nb_frames=7) scored F1=0.825 on val/all.
The sweep-optimized params improve F1 by ~9pp.

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
  -> infer (01_raw -> 03_primary)          # YOLO-only, cached per-frame detections
    -> predict (03_primary -> 07_model_output)  # replay temporal logic (fixed params)
      -> evaluate (-> 08_reporting)
    -> sweep (03_primary -> 08_reporting/sweep) # grid-search temporal params
  -> package (01_raw -> 06_models)
```

## Parameters

Current values are sweep-optimized. Production defaults from
[pyro-engine](https://github.com/pyronear/pyro-engine) are shown for
reference.

| Parameter | Value | Production | Description |
|---|---|---|---|
| `predict.conf_thresh` | **0.2** | 0.35 | **Alert threshold.** The Predictor accumulates detections across the sliding window and computes an aggregated confidence score. An alert fires when this score exceeds `conf_thresh`. Also used to filter individual detections before aggregation. |
| `predict.model_conf_thresh` | 0.05 | 0.05 | **YOLO detection threshold.** Minimum confidence for the YOLO model to report a bounding box. Kept low so the temporal layer sees all candidate detections. |
| `predict.nb_consecutive_frames` | **4** | 7 | **Sliding window size.** Number of recent frames the Predictor keeps in its per-camera state for temporal smoothing. A detection must persist across enough of these frames to trigger an alert. |
| `predict.max_bbox_size` | 0.4 | 0.4 | **Max detection width.** Detections wider than this fraction of the image are discarded (filters out large false positives like clouds). |

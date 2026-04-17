# 🏆 Temporal Model Leaderboard

Standardized evaluation and ranking of `TemporalModel` implementations on the [pyro-dataset](https://github.com/pyronear/pyro-dataset) **v3.0.0** sequential test set.

## 📊 Leaderboard

| Rank | Model | Precision | Recall | F1 | FPR | Mean TTD (s) | Median TTD (s) |
|------|-------|-----------|--------|----|-----|--------------|----------------|
| 1 | [FSM Tracking Baseline](../tracking-fsm-baseline/) | 0.9474 | 0.9664 | 0.9568 | 0.0537 | 142.0 | 58.0 |
| 2 | [Pyro-Detector Baseline](../pyro-detector-baseline/) | 0.8563 | 0.9597 | 0.9051 | 0.1611 | 27.0 | 7.0 |
| 3 | [Bbox-Tube Temporal](../bbox-tube-temporal/) | 0.8136 | 0.9664 | 0.8834 | 0.2215 | 503.1 | 328.5 |
| 4 | [MTB Change Detection](../mtb-change-detection/) | 0.7165 | 0.9329 | 0.8105 | 0.3691 | 85.4 | 25.0 |

*Evaluated on 298 sequences (149 wildfire + 149 false positive). Last updated: 2026-04-16.*

## 🤖 Models

| Model | Description | Paper |
|-------|-------------|-------|
| [FSM Tracking Baseline](../tracking-fsm-baseline/) | [YOLO11s `nimble-narwhal` v6.0.0](https://huggingface.co/pyronear/yolo11s_nimble-narwhal_v6.0.0) + IoU-based FSM tracker. Requires temporal persistence (5 consecutive frames) before raising an alarm. Rule-based, no ML training. | [FLAME (Gragnaniello et al., 2024)](https://doi.org/10.1007/s00521-024-10963-z) |
| [Pyro-Detector Baseline](../pyro-detector-baseline/) | Production pyro-predictor: [YOLO11s ONNX `nimble-narwhal` v6.0.0](https://huggingface.co/pyronear/yolo11s_nimble-narwhal_v6.0.0) + per-camera sliding-window temporal smoothing. Alarm when aggregated confidence crosses threshold over N consecutive frames. | -- |
| [MTB Change Detection](../mtb-change-detection/) | [YOLO11s `nimble-narwhal` v6.0.0](https://huggingface.co/pyronear/yolo11s_nimble-narwhal_v6.0.0) + pixel-wise frame differencing (MTB ratio) to reject static FPs, followed by IoU-based FSM tracker. | [SlowFastMTB (Choi, Kim & Oh, 2022)](https://doi.org/10.3390/s22155602) |
| [Bbox-Tube Temporal](../bbox-tube-temporal/) | YOLO11s companion + bbox-tube builder (IoU matching over ≤20 frames) + ViT-S/14 DINOv2 feature extractor with a transformer temporal head. Binary tube-level classifier with threshold calibrated to val recall=0.95. | -- |

## 📏 Metrics

- **Precision, Recall, F1** -- sequence-level binary classification (smoke vs. no smoke)
- **FPR** -- false positive rate
- **Mean / Median TTD** -- time-to-detection in seconds for true positives (time from first frame to trigger frame)

## 📦 Data

Test set imported via DVC from [pyro-dataset](https://github.com/pyronear/pyro-dataset) v3.0.0:
- 151 wildfire (positive) + 151 false positive (negative) sequences
- Ground truth determined by directory structure (`wildfire/` vs `fp/`)
- Max 20 frames per sequence, 30s apart

## 🔄 How to Reproduce

```bash
make install
uv run dvc pull            # pull test set + model packages from S3
uv run dvc repro           # run evaluation pipeline
```

## ➕ Adding a New Model

1. Implement `TemporalModel` in a new experiment under `experiments/temporal-models/`
2. Package the model (see [tracking-fsm-baseline](../tracking-fsm-baseline/) for the zip format)
3. Add a `dvc add` for the model package in `data/01_raw/models/`
4. Register the model in `MODEL_REGISTRY` in `scripts/evaluate.py`
5. Add an `evaluate_<name>` stage in `dvc.yaml` (with `--model-type <registry-key>`)
6. Run `uv run dvc repro`

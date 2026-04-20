# 🏆 Temporal Model Leaderboard

Standardized evaluation and ranking of `TemporalModel` implementations on the [pyro-dataset](https://github.com/pyronear/pyro-dataset) **v3.0.0** sequential test set.

## 📊 Leaderboard

| Rank | Model | Precision | Recall | F1 | FPR | Mean TTD (frames) | Median TTD (frames) |
|------|-------|-----------|--------|----|-----|-------------------|---------------------|
| 1 | [Bbox-Tube Temporal (ViT-DINOv2)](../bbox-tube-temporal/) | 0.9608 | 0.9735 | 0.9671 | 0.0397 | 3.4 | 2.0 |
| 2 | [FSM Tracking Baseline](../tracking-fsm-baseline/) | 0.9474 | 0.9536 | 0.9505 | 0.0530 | 4.6 | 4.0 |
| 3 | [Bbox-Tube Temporal (GRU-ConvNeXt)](../bbox-tube-temporal/) | 0.9272 | 0.9272 | 0.9272 | 0.0728 | 2.0 | 1.0 |
| 4 | [Pyro-Detector Baseline](../pyro-detector-baseline/) | 0.8580 | 0.9603 | 0.9063 | 0.1589 | 1.3 | 1.0 |
| 5 | [MTB Change Detection](../mtb-change-detection/) | 0.7121 | 0.9338 | 0.8080 | 0.3775 | 3.0 | 2.0 |

*Evaluated on 302 sequences (151 wildfire + 151 false positive). Last updated: 2026-04-20.*

## 🤖 Models

| Model | Description | Paper |
|-------|-------------|-------|
| [FSM Tracking Baseline](../tracking-fsm-baseline/) | [YOLO11s `nimble-narwhal` v6.0.0](https://huggingface.co/pyronear/yolo11s_nimble-narwhal_v6.0.0) + IoU-based FSM tracker. Requires temporal persistence (5 consecutive frames) before raising an alarm. Rule-based, no ML training. | [FLAME (Gragnaniello et al., 2024)](https://doi.org/10.1007/s00521-024-10963-z) |
| [Pyro-Detector Baseline](../pyro-detector-baseline/) | Production pyro-predictor: [YOLO11s ONNX `nimble-narwhal` v6.0.0](https://huggingface.co/pyronear/yolo11s_nimble-narwhal_v6.0.0) + per-camera sliding-window temporal smoothing. Alarm when aggregated confidence crosses threshold over N consecutive frames. | -- |
| [MTB Change Detection](../mtb-change-detection/) | [YOLO11s `nimble-narwhal` v6.0.0](https://huggingface.co/pyronear/yolo11s_nimble-narwhal_v6.0.0) + pixel-wise frame differencing (MTB ratio) to reject static FPs, followed by IoU-based FSM tracker. | [SlowFastMTB (Choi, Kim & Oh, 2022)](https://doi.org/10.3390/s22155602) |
| [Bbox-Tube Temporal](../bbox-tube-temporal/) | YOLO11s companion + bbox-tube builder (IoU matching over ≤20 frames) + ViT-S/14 DINOv2 (or GRU-ConvNeXt) feature extractor with a temporal head. Variant-specific decision rule: ViT uses a sequence-level multivariate logistic calibrator on `(logit, log-tube-length, mean-YOLO-conf, n-tubes)`; GRU uses raw-logit thresholding. Both threshold-calibrated to val recall=0.95. First-crossing trigger reports the earliest firing frame for accurate TTD. | -- |

## 📏 Metrics

- **Precision, Recall, F1** -- sequence-level binary classification (smoke vs. no smoke)
- **FPR** -- false positive rate
- **Mean / Median TTD** -- time-to-detection in **frames** for true positives (0-based `trigger_frame_index` where the model first decides positive). Frames are nominally 30s apart in production, but filename timestamps in the sequential test set are unreliable so we report in frames directly

## 📦 Data

Test set imported via DVC from [pyro-dataset](https://github.com/pyronear/pyro-dataset) v3.0.0:
- 151 wildfire (positive) + 151 false positive (negative) sequences
- Ground truth determined by directory structure (`wildfire/` vs `fp/`)
- Up to ~40 frames per sequence; production cadence is nominally 30s per frame, but filename timestamps in this test set are unreliable (see the TTD note above)

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

# ⏱️ Temporal Models

Experiments exploring temporal/video-based approaches to reduce false positives from the production YOLO smoke detector. Pyronear cameras capture frames every 30 seconds from fixed positions -- temporal models exploit this sequential structure to distinguish real smoke (which persists and grows) from transient false positives.

## 🧪 Experiments

| Experiment | Description | Paper |
|------------|-------------|-------|
| [tracking-fsm-baseline](tracking-fsm-baseline/) | YOLO11s detector + IoU-based FSM tracker. Requires temporal persistence (5 consecutive frames) before raising an alarm. Rule-based, no ML training. | [FLAME (Gragnaniello et al., 2024)](https://doi.org/10.1007/s00521-024-10963-z) |
| [mtb-change-detection](mtb-change-detection/) | YOLO detection + pixel-wise frame differencing (MTB) to reject static false positives. Detections must show real pixel-level change to be confirmed. | [SlowFastMTB (Choi et al., 2022)](https://doi.org/10.1093/jcde/qwac027) |
| [pyro-detector-baseline](pyro-detector-baseline/) | Production pyro-engine predictor (`pyro-predictor`) wrapped as a `TemporalModel` baseline. YOLO ONNX detection + per-camera sliding-window temporal smoothing. | -- |
| [bbox-tube-temporal](bbox-tube-temporal/) | Learned temporal smoke classifier over YOLO-linked bbox tubes. 224×224 context-expanded patches through a timm backbone + temporal head (mean-pool, GRU, or transformer) produce a sequence-level smoke/no-smoke logit. | -- |

## 🏆 Leaderboard

See [temporal-model-leaderboard](temporal-model-leaderboard/README.md) for full rankings and reproduction steps.

## 📚 Related Papers

See the [literature survey](../../literature_survey/SUMMARY.md) for temporal/video models for smoke detection.

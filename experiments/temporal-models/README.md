# Temporal Models

Experiments exploring temporal/video-based approaches to reduce false positives from the production YOLO smoke detector. Pyronear cameras capture frames every 30 seconds from fixed positions -- temporal models exploit this sequential structure to distinguish real smoke (which persists and grows) from transient false positives.

## Experiments

| Experiment | Description | Paper |
|------------|-------------|-------|
| [tracking-fsm-baseline](tracking-fsm-baseline/) | YOLO11s detector + IoU-based FSM tracker. Requires temporal persistence (5 consecutive frames) before raising an alarm. Rule-based, no ML training. | [FLAME (Gragnaniello et al., 2024)](https://doi.org/10.1007/s00521-024-10963-z) |
| [temporal-model-leaderboard](temporal-model-leaderboard/) | Standardized evaluation and ranking of `TemporalModel` implementations on the [pyro-dataset](https://github.com/pyronear/pyro-dataset) sequential test set. | -- |

## Leaderboard

Current rankings on the test set (298 sequences: 149 wildfire + 149 false positive):

| Rank | Model | Precision | Recall | F1 | FPR | Mean TTD | Median TTD |
|------|-------|-----------|--------|----|-----|----------|------------|
| 1 | [FSM Tracking Baseline](tracking-fsm-baseline/) | 0.9474 | 0.9664 | 0.9568 | 0.0537 | 142s | 58s |

See [temporal-model-leaderboard](temporal-model-leaderboard/) for full details and reproduction steps.

## Related Papers

Temporal/video models for smoke detection explored in the [literature survey](../../literature_survey/SUMMARY.md):

| Paper | Year | Approach | Link |
|-------|------|----------|------|
| SmokeyNet (Dewangan et al.) | 2022 | ResNet34 + LSTM + ViT spatiotemporal architecture | [arXiv](https://arxiv.org/abs/2112.08598) |
| SlowFastMTB (Choi et al.) | 2022 | SlowFast dual-pathway + MTB change detection | [DOI](https://doi.org/10.1093/jcde/qwac027) |
| Lightweight Student LSTM (Jeong et al.) | 2020 | YOLOv3 + distilled LSTM on smoke-tube features | [DOI](https://doi.org/10.3390/s20195508) |
| FLAME (Gragnaniello et al.) | 2024 | CNN detector + physics-informed motion filter | [DOI](https://doi.org/10.1007/s00521-024-10963-z) |
| Fire-Tube (Park & Ko) | 2020 | ELASTIC-YOLOv3 + temporal tube with optical flow | [DOI](https://doi.org/10.3390/s20082202) |
| ViT + 3D-CNN (Lilhore et al.) | 2026 | Vision Transformer with 3D-CNN spatiotemporal learning | [DOI](https://doi.org/10.1038/s41598-026-36687-9) |
| TeSTra (Zhao & Krahenbuhl) | 2022 | O(1) streaming detection via temporal smoothing kernels | [arXiv](https://arxiv.org/abs/2209.09236) |
| LSTR (Xu et al.) | 2021 | Dual long/short-term memory Transformer for online detection | [arXiv](https://arxiv.org/abs/2107.03377) |
| MATR (Song et al.) | 2024 | Memory-augmented Transformer for online temporal localization | [arXiv](https://arxiv.org/abs/2408.02957) |

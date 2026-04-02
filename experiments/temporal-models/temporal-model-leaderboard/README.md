# 🏆 Temporal Model Leaderboard

Standardized evaluation and ranking of `TemporalModel` implementations on the [pyro-dataset](https://github.com/pyronear/pyro-dataset) **v2.2.0** sequential test set.

## 📊 Leaderboard

| Rank | Model | Precision | Recall | F1 | FPR | Mean TTD (s) | Median TTD (s) |
|------|-------|-----------|--------|----|-----|--------------|----------------|
| 1 | [FSM Tracking Baseline](../tracking-fsm-baseline/) | 0.9474 | 0.9664 | 0.9568 | 0.0537 | 142.0 | 58.0 |

*Evaluated on 298 sequences (149 wildfire + 149 false positive). Last updated: 2026-03-31.*

## 🤖 Models

| Model | Description | Paper |
|-------|-------------|-------|
| [FSM Tracking Baseline](../tracking-fsm-baseline/) | YOLO11s detector + IoU-based FSM tracker. Requires temporal persistence (5 consecutive frames) before raising an alarm. Rule-based, no ML training. | [FLAME (Gragnaniello et al., 2024)](https://doi.org/10.1007/s00521-024-10963-z) |

## 📏 Metrics

- **Precision, Recall, F1** -- sequence-level binary classification (smoke vs. no smoke)
- **FPR** -- false positive rate
- **Mean / Median TTD** -- time-to-detection in seconds for true positives (time from first frame to trigger frame)

## 📦 Data

Test set imported via DVC from [pyro-dataset](https://github.com/pyronear/pyro-dataset):
- 149 wildfire (positive) + 149 false positive (negative) sequences
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

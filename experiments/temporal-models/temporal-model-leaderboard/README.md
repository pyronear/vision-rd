# Temporal Model Leaderboard

Evaluate and rank temporal smoke detection models on the pyro-dataset test set.

## Objective

Provide a standardized leaderboard comparing `TemporalModel` implementations
on the sequential test set from [pyro-dataset](https://github.com/pyronear/pyro-dataset) v2.2.0.

## Metrics

- **Precision, Recall, F1** (sequence-level classification)
- **FPR** (false positive rate)
- **Mean / Median TTD** (time-to-detection in seconds for true positives)

## Data

Test set imported via DVC from `pyro-dataset` v2.2.0:
- 149 wildfire (positive) + 149 false positive (negative) sequences
- Ground truth determined by directory (`wildfire/` vs `fp/`)

## How to Reproduce

```bash
make install
uv run dvc pull            # pull test set + model packages
uv run dvc repro           # run evaluation pipeline
```

## Results

See `data/08_reporting/leaderboard.json` and `data/08_reporting/leaderboard.txt`.

# Project Name

## Objective

_What problem does this project address?_

## Approach

_Method and architecture choices._

## Data

Imported from [pyro-dataset](https://github.com/pyronear/pyro-dataset) v2.2.0
via `dvc import` (sequential_train_val split):

- **Train**: _N_ wildfire + _N_ FP sequences
- **Val**: _N_ wildfire + _N_ FP sequences
- Layout: `data/01_raw/datasets/{train,val}/{wildfire,fp}/sequence_name/{images,labels}/`
- Ground truth: inferred from parent directory name (`wildfire/` = positive, `fp/` = negative)

## Results

_Key metrics and comparison to baselines._

## How to Reproduce

```bash
cd experiments/<category>/<experiment-name>
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

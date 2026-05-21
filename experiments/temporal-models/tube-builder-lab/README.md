# Tube Builder Lab

A standalone Streamlit lab to iterate on the **bbox-tube linking algorithm**.
For a set of real failure-case sequences it shows, side by side, the tubes the
**current** builder produces vs. the tubes an editable **candidate** builder
produces — so you can fix over-fragmentation and confirm it by eye.

Run every command from this directory:

```bash
cd experiments/temporal-models/tube-builder-lab
```

## Quick start

```bash
make install                         # uv sync + nbstripout
# (operator) sync the DVC-tracked sequences + detections, then:
make app                             # launch the lab at http://localhost:8501
```

## Pipeline

```
platform API ──import_sequences.py──> data/03_primary/sequences/<key>/   (frames)
model.zip   ──cache_detections.py───> data/05_model_input/detections/    (per-frame detections)
detections + candidate.py ──> app.py (current vs candidate, Layout A)
```

Iterate by editing `src/tube_builder_lab/candidate.py` and clicking **Re-run
candidate** in the app.

## Common commands

```bash
make install   # uv sync
make app       # launch the lab
make cache     # run YOLO + cache detections for the working set
make test      # pytest
make lint      # ruff check
make format    # ruff format
```

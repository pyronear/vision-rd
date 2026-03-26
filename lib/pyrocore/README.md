# pyrocore

Shared types and base model for Pyronear temporal smoke detection experiments.

## What's inside

- `TemporalModel` — ABC that all temporal experiments implement (template method: `load_sequence` → `predict`)
- `Frame` — minimal frame representation (frame_id, image_path, optional timestamp)
- `TemporalModelOutput` — standardized prediction output (is_positive, trigger_frame_index, details)

## Usage

Experiments add pyrocore as a path dependency in their `pyproject.toml`:

```toml
dependencies = [
    "pyrocore @ file:///${PROJECT_ROOT}/../../lib/pyrocore",
]
```

Then subclass `TemporalModel` and implement `predict`:

```python
from pyrocore import TemporalModel, Frame, TemporalModelOutput


class MyModel(TemporalModel):
    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        # model logic here
        return TemporalModelOutput(is_positive=True, trigger_frame_index=4)


model = MyModel()
output = model.predict_sequence(sorted_frame_paths)
```

Override `load_sequence` to customize how image paths are converted to `Frame` objects (e.g., parse timestamps differently, load cached detections).

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

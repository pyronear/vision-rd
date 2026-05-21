# Tube Builder Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, example-driven Streamlit lab that shows the current vs. a candidate smoke-tube builder side by side on real failure-case sequences, so the linking algorithm can be iterated and visually verified.

**Architecture:** A new isolated uv experiment. An offline `import_sequences` script populates the experiment's own DVC-tracked sequence store (by sequence id); an offline `cache_detections` script runs the model's bundled YOLO once and serializes per-frame detections + the pipeline config. The Streamlit app reads only that cached data: it builds "current" tubes (lib `build_tubes`) and "candidate" tubes (an editable, hot-reloaded `candidate.py`), passes both through the same filter, and renders Layout A (shared frame player + two stacked tube timelines on one frame axis).

**Tech Stack:** Python 3.11, uv, pytest, ruff, Streamlit + Altair, Pillow, PyYAML, pandas; depends on `bbox-tube-temporal-core` and `pyrocore` (path sources); DVC (S3) for data.

Spec: [`docs/specs/2026-05-21-tube-builder-lab-design.md`](../specs/2026-05-21-tube-builder-lab-design.md)

**Faithful config values** (read from `data/06_models/vit_dinov2_finetune/model.zip::config.yaml`, used throughout this plan):
- detections: `confidence_threshold=0.1`, `iou_nms=0.2`, `image_size=1024`
- tubes: `iou_threshold=0.2`, `max_misses=2`, `infer_min_tube_length=2`, `min_detected_entries=2`, `interpolate_gaps=true`
- truncation: `max_frames=20`

> **Operator notes (apply to every task):**
> - Run all commands from `experiments/temporal-models/tube-builder-lab/`.
> - Stage files **explicitly** by path in every commit (never `git add -A` / globs).
> - Do **not** add Claude/Anthropic co-author trailers to commits.
> - Do **not** run `dvc pull`. DVC pull/push and platform imports are operator-run steps (Task 11), called out explicitly.

---

## Task 1: Scaffold the isolated experiment

**Files:**
- Create (copy): the whole `experiments/temporal-models/tube-builder-lab/` tree from `experiments/template/`
- Modify: `pyproject.toml`
- Modify: `Makefile`
- Create: `README.md`
- Rename: `src/project_name/` → `src/tube_builder_lab/`

- [ ] **Step 1: Copy the template into place** (the spec/plan dirs already exist under the target; copy the rest)

```bash
cd experiments/temporal-models
cp -rn template/. tube-builder-lab/
cd tube-builder-lab
git mv src/project_name src/tube_builder_lab 2>/dev/null || mv src/project_name src/tube_builder_lab
```

- [ ] **Step 2: Replace `pyproject.toml`**

```toml
[project]
name = "tube-builder-lab"
version = "0.1.0"
description = "Visual lab to iterate on the bbox-tube linking algorithm (current vs candidate)"
requires-python = ">=3.11"
dependencies = [
    "bbox-tube-temporal-core",
    "pyrocore",
    "requests>=2.31",
    "pyyaml>=6.0",
    "pillow>=10.0",
    "pandas>=2.0",
    "altair>=5.0",
    "streamlit>=1.40",
]

[tool.uv.sources]
bbox-tube-temporal-core = { path = "../../../lib/bbox-tube-temporal" }
pyrocore = { path = "../../../lib/pyrocore" }

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.9", "dvc[s3]>=3.56"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tube_builder_lab"]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 3: Add `app` and `cache` targets to `Makefile`** (append after the `notebook` target)

```makefile
app: ## Launch the Streamlit tube lab
	uv run streamlit run src/tube_builder_lab/app.py

cache: ## Run YOLO over the working-set sequences and cache detections
	uv run python scripts/cache_detections.py
```

Also add `app cache` to the `.PHONY` line.

- [ ] **Step 4: Write `README.md`** (do NOT reference the spec file in the README)

```markdown
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
```

- [ ] **Step 5: Replace the template smoke test** so the package imports

`tests/test_smoke.py`:

```python
def test_package_imports():
    import tube_builder_lab  # noqa: F401
```

- [ ] **Step 6: Install and verify the scaffold**

Run: `make install && make test`
Expected: install succeeds; `tests/test_smoke.py::test_package_imports` PASSES.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml Makefile README.md src/tube_builder_lab/__init__.py tests/test_smoke.py
git commit -m "feat(tube-lab): scaffold isolated experiment"
```

---

## Task 2: Sequence store (flat layout)

**Files:**
- Create: `src/tube_builder_lab/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:

```python
from pathlib import Path

from tube_builder_lab.store import (
    FrameRef,
    SequenceMeta,
    build_frames,
    iter_sequence_dirs,
    read_meta,
    seq_dir_for_key,
    write_meta,
)


def test_meta_roundtrip_and_lookup(tmp_path: Path):
    store = tmp_path / "sequences"
    meta = SequenceMeta(
        key="platform_42",
        sequence_id="42",
        frames=[
            FrameRef(file="images/a.jpg", detection_id=1, created_at="2026-05-17T10:00:00"),
            FrameRef(file="images/b.jpg", detection_id=2, created_at="2026-05-17T10:00:30"),
        ],
    )
    seq_dir = store / "platform_42"
    write_meta(seq_dir, meta)

    assert [d.name for d in iter_sequence_dirs(store)] == ["platform_42"]
    assert seq_dir_for_key(store, "platform_42") == seq_dir
    assert seq_dir_for_key(store, "nope") is None

    got = read_meta(seq_dir)
    assert got == meta

    frames = build_frames(seq_dir, got)
    assert [f.frame_id for f in frames] == ["a", "b"]
    assert frames[0].image_path == seq_dir / "images/a.jpg"
    assert frames[1].timestamp is not None
```

- [ ] **Step 2: Run; expect failure**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: tube_builder_lab.store`.

- [ ] **Step 3: Implement `store.py`**

```python
"""Local sequence store (flat `<key>/` layout): meta IO + Frame helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from pyrocore import Frame

META_FILENAME = "meta.json"


@dataclass
class FrameRef:
    file: str  # relative to the sequence dir, e.g. "images/detection_5.jpg"
    detection_id: int | None = None
    created_at: str | None = None  # ISO timestamp


@dataclass
class SequenceMeta:
    key: str
    sequence_id: str
    frames: list[FrameRef] = field(default_factory=list)


def write_meta(seq_dir: Path, meta: SequenceMeta) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / META_FILENAME).write_text(json.dumps(asdict(meta), indent=2))


def read_meta(seq_dir: Path) -> SequenceMeta:
    payload = json.loads((seq_dir / META_FILENAME).read_text())
    frames = [FrameRef(**f) for f in payload.pop("frames", [])]
    return SequenceMeta(frames=frames, **payload)


def iter_sequence_dirs(store_dir: Path) -> Iterator[Path]:
    """Yield every directory under ``store_dir`` containing a meta.json."""
    if not store_dir.exists():
        return
    for meta_path in sorted(store_dir.rglob(META_FILENAME)):
        yield meta_path.parent


def seq_dir_for_key(store_dir: Path, key: str) -> Path | None:
    """Resolve a sequence dir by its meta key (flat layout = store/<key>)."""
    direct = store_dir / key
    if (direct / META_FILENAME).exists():
        return direct
    for seq_dir in iter_sequence_dirs(store_dir):
        if read_meta(seq_dir).key == key:
            return seq_dir
    return None


def build_frames(seq_dir: Path, meta: SequenceMeta) -> list[Frame]:
    """Ordered pyrocore Frames; meta order is the time axis."""
    frames: list[Frame] = []
    for ref in meta.frames:
        ts = None
        if ref.created_at:
            try:
                ts = datetime.fromisoformat(ref.created_at.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        frames.append(
            Frame(
                frame_id=Path(ref.file).stem,
                image_path=seq_dir / ref.file,
                timestamp=ts,
            )
        )
    return frames
```

- [ ] **Step 4: Run; expect pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tube_builder_lab/store.py tests/test_store.py
git commit -m "feat(tube-lab): sequence store (flat layout)"
```

---

## Task 3: Detections IO (cache round-trip)

**Files:**
- Create: `src/tube_builder_lab/detections_io.py`
- Test: `tests/test_detections_io.py`

- [ ] **Step 1: Write the failing test**

`tests/test_detections_io.py`:

```python
from pathlib import Path

from bbox_tube_temporal.types import Detection, FrameDetections
from tube_builder_lab.detections_io import read_detections, write_detections


def test_detections_roundtrip(tmp_path: Path):
    fds = [
        FrameDetections(
            frame_idx=0,
            frame_id="a",
            timestamp=None,
            detections=[
                Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.2, confidence=0.9),
                Detection(class_id=0, cx=0.2, cy=0.3, w=0.05, h=0.05, confidence=0.4),
            ],
        ),
        FrameDetections(frame_idx=1, frame_id="b", timestamp=None, detections=[]),
    ]
    path = tmp_path / "platform_42.json"
    write_detections(path, fds)
    got = read_detections(path)
    assert got == fds
```

- [ ] **Step 2: Run; expect failure**

Run: `uv run pytest tests/test_detections_io.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `detections_io.py`**

```python
"""Serialize per-frame YOLO detections to/from the on-disk cache."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from bbox_tube_temporal.types import Detection, FrameDetections


def _det_to_dict(d: Detection) -> dict:
    return {
        "class_id": d.class_id,
        "cx": d.cx,
        "cy": d.cy,
        "w": d.w,
        "h": d.h,
        "confidence": d.confidence,
    }


def _det_from_dict(o: dict) -> Detection:
    return Detection(
        class_id=o["class_id"],
        cx=o["cx"],
        cy=o["cy"],
        w=o["w"],
        h=o["h"],
        confidence=o["confidence"],
    )


def _fd_to_dict(fd: FrameDetections) -> dict:
    return {
        "frame_idx": fd.frame_idx,
        "frame_id": fd.frame_id,
        "timestamp": fd.timestamp.isoformat() if fd.timestamp else None,
        "detections": [_det_to_dict(d) for d in fd.detections],
    }


def _fd_from_dict(o: dict) -> FrameDetections:
    ts = o.get("timestamp")
    return FrameDetections(
        frame_idx=o["frame_idx"],
        frame_id=o["frame_id"],
        timestamp=datetime.fromisoformat(ts) if ts else None,
        detections=[_det_from_dict(d) for d in o["detections"]],
    )


def write_detections(path: Path, frame_detections: list[FrameDetections]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_fd_to_dict(fd) for fd in frame_detections], indent=2))


def read_detections(path: Path) -> list[FrameDetections]:
    return [_fd_from_dict(o) for o in json.loads(path.read_text())]
```

- [ ] **Step 4: Run; expect pass**

Run: `uv run pytest tests/test_detections_io.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tube_builder_lab/detections_io.py tests/test_detections_io.py
git commit -m "feat(tube-lab): detections cache IO"
```

---

## Task 4: Working-set loader

**Files:**
- Create: `src/tube_builder_lab/working_set.py`
- Create: `working_set.yaml`
- Test: `tests/test_working_set.py`

- [ ] **Step 1: Write the failing test**

`tests/test_working_set.py`:

```python
from pathlib import Path

from tube_builder_lab.working_set import WorkingItem, load_working_set


def test_load_working_set(tmp_path: Path):
    p = tmp_path / "ws.yaml"
    p.write_text(
        "targets:\n"
        "  - { key: platform_1, note: 'three into one' }\n"
        "  - { key: platform_2 }\n"
        "control:\n"
        "  - { key: platform_9 }\n"
    )
    ws = load_working_set(p)
    assert ws.targets == [
        WorkingItem(key="platform_1", note="three into one"),
        WorkingItem(key="platform_2", note=None),
    ]
    assert ws.control == [WorkingItem(key="platform_9", note=None)]
    assert [i.key for i in ws.all()] == ["platform_1", "platform_2", "platform_9"]


def test_load_working_set_empty_control(tmp_path: Path):
    p = tmp_path / "ws.yaml"
    p.write_text("targets:\n  - { key: platform_1 }\ncontrol: []\n")
    ws = load_working_set(p)
    assert ws.control == []
```

- [ ] **Step 2: Run; expect failure**

Run: `uv run pytest tests/test_working_set.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `working_set.py`**

```python
"""Load the curated working set (targets + control) from working_set.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class WorkingItem:
    key: str
    note: str | None = None


@dataclass
class WorkingSet:
    targets: list[WorkingItem]
    control: list[WorkingItem]

    def all(self) -> list[WorkingItem]:
        return [*self.targets, *self.control]


def _items(raw: list | None) -> list[WorkingItem]:
    return [WorkingItem(key=o["key"], note=o.get("note")) for o in (raw or [])]


def load_working_set(path: Path) -> WorkingSet:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return WorkingSet(
        targets=_items(payload.get("targets")),
        control=_items(payload.get("control")),
    )
```

- [ ] **Step 4: Create `working_set.yaml`** (the collected failure cases; `control` filled later)

```yaml
targets:
  - { key: platform_43096 }
  - { key: platform_42466 }
  - { key: platform_41304, note: "the three tubes should only be one" }
  - { key: platform_41319 }
  - { key: platform_41310, note: "T3 and T4 should be the same tube?" }
  - { key: platform_41289, note: "T0 and T3 should be the same tube" }
  - { key: platform_41209 }
  - { key: platform_40800 }
  - { key: platform_41616, note: "the three tubes should only be one" }
  - { key: platform_41786, note: "the tubes could be merged into one" }
  - { key: platform_42562 }
  - { key: platform_42538, note: "should be only one tube" }
  - { key: platform_41887 }
  - { key: platform_41541 }
  - { key: platform_43206, note: "T0 and T5 should be one; T1 and T3 should be one" }
  - { key: platform_42910 }
control: []
```

- [ ] **Step 5: Run; expect pass**

Run: `uv run pytest tests/test_working_set.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add src/tube_builder_lab/working_set.py working_set.yaml tests/test_working_set.py
git commit -m "feat(tube-lab): working-set loader + collected cases"
```

---

## Task 5: Pipeline (config + comparable two-sides)

**Files:**
- Create: `src/tube_builder_lab/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:

```python
from bbox_tube_temporal.types import Detection, FrameDetections
from tube_builder_lab.pipeline import (
    PipelineConfig,
    current_builder,
    detections_to_display_tubes,
    extract_pipeline_config,
)


def _fd(idx, dets):
    return FrameDetections(frame_idx=idx, frame_id=str(idx), timestamp=None, detections=dets)


def _d(cx, cy):
    return Detection(class_id=0, cx=cx, cy=cy, w=0.1, h=0.1, confidence=0.9)


CFG = PipelineConfig(
    max_frames=20,
    iou_threshold=0.2,
    max_misses=2,
    infer_min_tube_length=2,
    min_detected_entries=2,
    interpolate_gaps=True,
    confidence_threshold=0.1,
    iou_nms=0.2,
    image_size=1024,
)


def test_extract_pipeline_config_from_model_config():
    raw = {
        "infer": {"confidence_threshold": 0.1, "iou_nms": 0.2, "image_size": 1024},
        "tubes": {
            "iou_threshold": 0.2,
            "max_misses": 2,
            "infer_min_tube_length": 2,
            "min_detected_entries": 2,
            "interpolate_gaps": True,
        },
        "classifier": {"max_frames": 20},
    }
    assert extract_pipeline_config(raw) == CFG


def test_current_builder_links_a_steady_box():
    # Same box across 3 frames -> a single kept tube (length 3 >= 2).
    fds = [_fd(i, [_d(0.5, 0.5)]) for i in range(3)]
    tubes = detections_to_display_tubes(fds, current_builder(CFG), CFG, truncate=True)
    assert len(tubes) == 1
    assert tubes[0].start_frame == 0
    assert tubes[0].end_frame == 2


def test_truncation_limits_frames():
    # 25 frames of a steady box; truncate to max_frames=20 -> tube ends at 19.
    fds = [_fd(i, [_d(0.5, 0.5)]) for i in range(25)]
    tubes = detections_to_display_tubes(fds, current_builder(CFG), CFG, truncate=True)
    assert len(tubes) == 1
    assert tubes[0].end_frame == 19
    # Untruncated -> ends at 24.
    tubes_full = detections_to_display_tubes(fds, current_builder(CFG), CFG, truncate=False)
    assert tubes_full[0].end_frame == 24


def test_filter_drops_singleton_tube():
    # A box present in only one frame -> length 1 < infer_min_tube_length -> dropped.
    fds = [_fd(0, [_d(0.5, 0.5)]), _fd(1, []), _fd(2, []), _fd(3, [])]
    tubes = detections_to_display_tubes(fds, current_builder(CFG), CFG, truncate=True)
    assert tubes == []
```

- [ ] **Step 2: Run; expect failure**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `pipeline.py`**

```python
"""Turn cached per-frame detections into comparable display tubes.

Both the current and the candidate builders run through the SAME truncation
and the SAME filter, so the only difference between the two sides is the
linking logic itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from bbox_tube_temporal.inference import filter_and_interpolate_tubes
from bbox_tube_temporal.tubes import build_tubes
from bbox_tube_temporal.types import FrameDetections, Tube

Builder = Callable[[list[FrameDetections]], list[Tube]]


@dataclass(frozen=True)
class PipelineConfig:
    max_frames: int
    iou_threshold: float
    max_misses: int
    infer_min_tube_length: int
    min_detected_entries: int
    interpolate_gaps: bool
    confidence_threshold: float
    iou_nms: float
    image_size: int


def extract_pipeline_config(model_config: dict) -> PipelineConfig:
    """Pull the lab-relevant knobs out of a packaged model config.yaml dict."""
    infer = model_config["infer"]
    tubes = model_config["tubes"]
    return PipelineConfig(
        max_frames=int(model_config["classifier"]["max_frames"]),
        iou_threshold=float(tubes["iou_threshold"]),
        max_misses=int(tubes["max_misses"]),
        infer_min_tube_length=int(tubes["infer_min_tube_length"]),
        min_detected_entries=int(tubes["min_detected_entries"]),
        interpolate_gaps=bool(tubes["interpolate_gaps"]),
        confidence_threshold=float(infer["confidence_threshold"]),
        iou_nms=float(infer["iou_nms"]),
        image_size=int(infer["image_size"]),
    )


def write_pipeline_config(path: Path, cfg: PipelineConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.__dict__, default_flow_style=False))


def load_pipeline_config(path: Path) -> PipelineConfig:
    return PipelineConfig(**yaml.safe_load(Path(path).read_text()))


def current_builder(cfg: PipelineConfig) -> Builder:
    """The current lib builder, bound to the model's tube params."""

    def _build(frame_detections: list[FrameDetections]) -> list[Tube]:
        return build_tubes(
            frame_detections,
            iou_threshold=cfg.iou_threshold,
            max_misses=cfg.max_misses,
        )

    return _build


def detections_to_display_tubes(
    frame_detections: list[FrameDetections],
    builder: Builder,
    cfg: PipelineConfig,
    *,
    truncate: bool,
) -> list[Tube]:
    fds = frame_detections[: cfg.max_frames] if truncate else frame_detections
    tubes = builder(fds)
    return filter_and_interpolate_tubes(
        tubes,
        min_tube_length=cfg.infer_min_tube_length,
        min_detected_entries=cfg.min_detected_entries,
        interpolate_gaps=cfg.interpolate_gaps,
    )
```

- [ ] **Step 4: Run; expect pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tube_builder_lab/pipeline.py tests/test_pipeline.py
git commit -m "feat(tube-lab): pipeline config + comparable two-sides"
```

---

## Task 6: Candidate builder (seed = current)

**Files:**
- Create: `src/tube_builder_lab/candidate.py`
- Test: `tests/test_candidate.py`

- [ ] **Step 1: Write the failing test** (candidate must equal the current builder on day one)

`tests/test_candidate.py`:

```python
from bbox_tube_temporal.tubes import build_tubes
from bbox_tube_temporal.types import Detection, FrameDetections
from tube_builder_lab.candidate import build_tubes_candidate


def _fd(idx, *boxes):
    return FrameDetections(
        frame_idx=idx,
        frame_id=str(idx),
        timestamp=None,
        detections=[Detection(class_id=0, cx=cx, cy=cy, w=0.1, h=0.1, confidence=0.9) for cx, cy in boxes],
    )


def test_candidate_seed_matches_current_builder():
    fds = [_fd(0, (0.5, 0.5)), _fd(1, (0.52, 0.5)), _fd(2, (0.8, 0.2))]
    expected = build_tubes(fds, iou_threshold=0.2, max_misses=2)
    got = build_tubes_candidate(fds)
    assert [(t.start_frame, t.end_frame, len(t.entries)) for t in got] == [
        (t.start_frame, t.end_frame, len(t.entries)) for t in expected
    ]
```

- [ ] **Step 2: Run; expect failure**

Run: `uv run pytest tests/test_candidate.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `candidate.py`** (the file we will edit during iteration)

```python
"""The candidate tube builder — EDIT THIS to improve linking.

Seeded identical to the current lib builder so the lab's diff starts empty.
Iterate freely here (containment / IoMin association, larger max_misses, a
post-hoc merge pass, ...) and click "Re-run candidate" in the app to see the
result against the working-set sequences. Build on the lib primitives
(`compute_iou`, `match_detections`, types) so propagating the winner to
lib/bbox-tube-temporal later is mechanical.
"""

from __future__ import annotations

from bbox_tube_temporal.tubes import build_tubes
from bbox_tube_temporal.types import FrameDetections, Tube

# Current defaults (kept in sync with the model config); change as you iterate.
IOU_THRESHOLD = 0.2
MAX_MISSES = 2


def build_tubes_candidate(frame_detections: list[FrameDetections]) -> list[Tube]:
    return build_tubes(
        frame_detections,
        iou_threshold=IOU_THRESHOLD,
        max_misses=MAX_MISSES,
    )
```

- [ ] **Step 4: Run; expect pass**

Run: `uv run pytest tests/test_candidate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tube_builder_lab/candidate.py tests/test_candidate.py
git commit -m "feat(tube-lab): candidate builder seeded as current"
```

---

## Task 7: Viz helpers (pure parts tested)

**Files:**
- Create: `src/tube_builder_lab/viz.py`
- Test: `tests/test_viz.py`

- [ ] **Step 1: Write the failing test**

`tests/test_viz.py`:

```python
from bbox_tube_temporal.types import Detection, Tube, TubeEntry
from tube_builder_lab.viz import (
    bboxes_at_frame,
    norm_bbox_to_pixel,
    tube_color,
    tube_timeline_df,
)


def _entry(idx, cx, cy, conf=0.9, gap=False):
    det = Detection(class_id=0, cx=cx, cy=cy, w=0.2, h=0.4, confidence=conf)
    return TubeEntry(frame_idx=idx, detection=det, is_gap=gap)


def _tube(tid, entries):
    return Tube(
        tube_id=tid,
        entries=entries,
        start_frame=entries[0].frame_idx,
        end_frame=entries[-1].frame_idx,
    )


def test_tube_color_is_stable_and_cyclic():
    assert tube_color(0) == tube_color(10)  # 10-colour palette
    assert tube_color(0) != tube_color(1)


def test_norm_bbox_to_pixel():
    # cx,cy,w,h normalized -> (x0,y0,x1,y1) pixels for a 100x200 image.
    assert norm_bbox_to_pixel((0.5, 0.5, 0.2, 0.4), 100, 200) == (40.0, 60.0, 60.0, 140.0)


def test_timeline_df_one_row_per_entry():
    t0 = _tube(0, [_entry(0, 0.5, 0.5), _entry(1, 0.5, 0.5, gap=True)])
    t1 = _tube(1, [_entry(2, 0.2, 0.2)])
    df = tube_timeline_df([t0, t1])
    assert list(df.columns) == ["tube", "frame", "frame_end", "confidence", "is_gap"]
    assert len(df) == 3
    assert set(df["tube"]) == {"T0", "T1"}
    assert df[df["frame"] == 1]["is_gap"].iloc[0]  # the gap entry is flagged


def test_bboxes_at_frame_picks_the_right_entry():
    t0 = _tube(0, [_entry(0, 0.5, 0.5), _entry(1, 0.6, 0.5)])
    t1 = _tube(1, [_entry(1, 0.2, 0.2)])
    got = bboxes_at_frame([t0, t1], 1)
    # (bbox, confidence, tube_id, is_gap) for every tube active at frame 1
    ids = sorted(g[2] for g in got)
    assert ids == [0, 1]
    box_t0 = next(g for g in got if g[2] == 0)[0]
    assert box_t0[0] == 0.6  # cx of t0 at frame 1
```

- [ ] **Step 2: Run; expect failure**

Run: `uv run pytest tests/test_viz.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `viz.py`**

```python
"""Pure viz helpers (timeline shaping, bbox geometry, colours) + PIL wrappers.

Operates directly on lib ``Tube`` objects. The pure helpers are unit-tested;
the PIL/crop wrappers are exercised by the app and manual verification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from bbox_tube_temporal.model_input import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
from bbox_tube_temporal.types import Tube
from PIL import Image, ImageDraw, ImageFont

CROP_CONTEXT = 2.0
CROP_SIZE = 224

TUBE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

try:
    _BBOX_FONT = ImageFont.load_default(size=18)
except TypeError:  # older Pillow without the size kwarg
    _BBOX_FONT = ImageFont.load_default()


def tube_color(tube_id: int) -> str:
    """Stable colour for a tube id (cycles through a 10-colour palette)."""
    return TUBE_PALETTE[tube_id % len(TUBE_PALETTE)]


def tube_count(tubes: list[Tube]) -> int:
    return len(tubes)


def norm_bbox_to_pixel(
    bbox: tuple[float, float, float, float], w: int, h: int
) -> tuple[float, float, float, float]:
    """(cx,cy,w,h) normalized -> (x0,y0,x1,y1) in pixels."""
    cx, cy, bw, bh = bbox
    return (
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    )


def bboxes_at_frame(
    tubes: list[Tube], frame_idx: int
) -> list[tuple[tuple[float, float, float, float], float, int, bool]]:
    """For each tube active at ``frame_idx``: (bbox, confidence, tube_id, is_gap)."""
    out = []
    for tube in tubes:
        for e in tube.entries:
            if e.frame_idx == frame_idx and e.detection is not None:
                d = e.detection
                out.append(((d.cx, d.cy, d.w, d.h), d.confidence, tube.tube_id, e.is_gap))
                break
    return out


def tube_timeline_df(tubes: list[Tube]) -> pd.DataFrame:
    """Long frame for the Altair timeline: one row per present tube entry."""
    records = [
        {
            "tube": f"T{tube.tube_id}",
            "frame": e.frame_idx,
            "frame_end": e.frame_idx + 1,
            "confidence": e.detection.confidence,
            "is_gap": e.is_gap,
        }
        for tube in tubes
        for e in tube.entries
        if e.detection is not None
    ]
    return pd.DataFrame(
        records, columns=["tube", "frame", "frame_end", "confidence", "is_gap"]
    )


def draw_tube_bboxes(image_path: Path, tubes: list[Tube], frame_idx: int, width: int = 4):
    """Frame image with each active tube's bbox drawn in its tube colour."""
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)
    for bbox, conf, tid, is_gap in bboxes_at_frame(tubes, frame_idx):
        x0, y0, x1, y1 = norm_bbox_to_pixel(bbox, w_img, h_img)
        color = tube_color(tid)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        label = f"T{tid}" + (" (gap)" if is_gap else f" {conf:.2f}")
        draw.text((x0, max(0, y0 - 20)), label, fill=color, font=_BBOX_FONT)
    return img


def crop_tube_at_frame(image_path: Path, bbox: tuple[float, float, float, float]):
    """Square context crop centred on a normalized bbox (matches the explorer)."""
    img = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w = img.shape[:2]
    cx, cy, bw, bh = bbox
    ecx, ecy, ew, eh = expand_bbox(cx, cy, bw, bh, CROP_CONTEXT)
    box = norm_bbox_to_pixel_square(ecx, ecy, ew, eh, img_w, img_h)
    return Image.fromarray(crop_and_resize(img, box, CROP_SIZE))
```

- [ ] **Step 4: Run; expect pass**

Run: `uv run pytest tests/test_viz.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tube_builder_lab/viz.py tests/test_viz.py
git commit -m "feat(tube-lab): viz helpers over Tube objects"
```

---

## Task 8: Platform client + import-by-id

**Files:**
- Create: `src/tube_builder_lab/platform_api.py`
- Create: `src/tube_builder_lab/import_sequences.py`
- Create: `scripts/import_sequences.py`
- Test: `tests/test_import_sequences.py`

- [ ] **Step 1: Write the failing test** (pure import-one with canned detections + stub downloader)

`tests/test_import_sequences.py`:

```python
from pathlib import Path

from tube_builder_lab.import_sequences import import_one_by_id
from tube_builder_lab.store import read_meta


def test_import_one_orders_frames_and_writes_meta(tmp_path: Path):
    store = tmp_path / "sequences"
    detections = [
        {"id": 20, "url": "http://x/2.jpg", "created_at": "2026-05-17T10:00:30"},
        {"id": 10, "url": "http://x/1.jpg", "created_at": "2026-05-17T10:00:00"},
        {"id": 30, "url": "http://x/3.jpg", "created_at": None},  # no ts -> sorts last-ish
    ]
    downloaded: list[str] = []

    def fake_download(url: str) -> bytes:
        downloaded.append(url)
        return b"jpeg-bytes"

    seq_dir = import_one_by_id(
        store_dir=store, sequence_id=42, detections=detections, download=fake_download
    )

    meta = read_meta(seq_dir)
    assert meta.key == "platform_42"
    assert meta.sequence_id == "42"
    # ordered by created_at ascending (None treated as empty string -> first)
    assert [f.detection_id for f in meta.frames] == [30, 10, 20]
    for f in meta.frames:
        assert (seq_dir / f.file).read_bytes() == b"jpeg-bytes"
    assert len(downloaded) == 3


def test_import_one_skips_detection_without_url(tmp_path: Path):
    store = tmp_path / "sequences"
    detections = [
        {"id": 1, "url": None, "created_at": "2026-05-17T10:00:00"},
        {"id": 2, "url": "http://x/2.jpg", "created_at": "2026-05-17T10:00:30"},
    ]
    seq_dir = import_one_by_id(
        store_dir=store, sequence_id=7, detections=detections, download=lambda u: b"x"
    )
    meta = read_meta(seq_dir)
    assert [f.detection_id for f in meta.frames] == [2]
```

- [ ] **Step 2: Run; expect failure**

Run: `uv run pytest tests/test_import_sequences.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the lean platform client `platform_api.py`** (duplicated by design, for isolation)

```python
"""Lean Pyronear platform API client — only the read endpoints this lab needs.

Duplicated (not shared) so the experiment stays fully isolated. Auth via
/login/creds, bearer header.
"""

from __future__ import annotations

import requests


def get_access_token(api_endpoint: str, username: str, password: str) -> str:
    resp = requests.post(
        f"{api_endpoint}/api/v1/login/creds",
        data={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def list_sequence_detections(
    api_endpoint: str, token: str, sequence_id: int, limit: int = 30
) -> list[dict]:
    route = (
        f"{api_endpoint}/api/v1/sequences/{sequence_id}/detections"
        f"?limit={limit}&desc=false"
    )
    resp = requests.get(route, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_image(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content
```

- [ ] **Step 4: Implement `import_sequences.py`** (the pure import-one + a thin orchestrator)

```python
"""Import working-set sequences BY ID into the lab's own store (flat layout)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from . import platform_api
from .store import FrameRef, SequenceMeta, write_meta

log = logging.getLogger(__name__)


def import_one_by_id(
    *,
    store_dir: Path,
    sequence_id: int,
    detections: list[dict],
    download: Callable[[str], bytes],
) -> Path:
    """Write frames + minimal meta for one sequence; returns its dir."""
    ordered = sorted(detections, key=lambda d: d.get("created_at") or "")
    seq_dir = store_dir / f"platform_{sequence_id}"
    (seq_dir / "images").mkdir(parents=True, exist_ok=True)

    frames: list[FrameRef] = []
    for det in ordered:
        url = det.get("url")
        if not url:
            log.warning("detection %s of seq %s has no url; skipping", det.get("id"), sequence_id)
            continue
        fname = f"detection_{det['id']}.jpg"
        (seq_dir / "images" / fname).write_bytes(download(url))
        frames.append(
            FrameRef(file=f"images/{fname}", detection_id=det["id"], created_at=det.get("created_at"))
        )

    write_meta(
        seq_dir,
        SequenceMeta(key=f"platform_{sequence_id}", sequence_id=str(sequence_id), frames=frames),
    )
    return seq_dir


def sequence_id_from_key(key: str) -> int:
    """'platform_42538' -> 42538."""
    return int(key.rsplit("_", 1)[-1])


def import_keys(
    *,
    store_dir: Path,
    keys: list[str],
    api_endpoint: str,
    token: str,
    detections_limit: int,
    download: Callable[[str], bytes] = platform_api.download_image,
) -> int:
    """Fetch + import every key from the platform. Returns #sequences imported."""
    count = 0
    for key in keys:
        sid = sequence_id_from_key(key)
        dets = platform_api.list_sequence_detections(
            api_endpoint, token, sid, limit=detections_limit
        )
        import_one_by_id(store_dir=store_dir, sequence_id=sid, detections=dets, download=download)
        count += 1
        log.info("imported %s (%d detections)", key, len(dets))
    return count
```

- [ ] **Step 5: Implement the CLI `scripts/import_sequences.py`**

```python
"""CLI: import the working-set sequences by id into the lab store.

Creds via env: PLATFORM_API_ENDPOINT, PLATFORM_LOGIN, PLATFORM_PASSWORD.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from tube_builder_lab import platform_api
from tube_builder_lab.import_sequences import import_keys
from tube_builder_lab.working_set import load_working_set

logging.basicConfig(level=logging.INFO)


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    store_dir = Path(params["store"])
    detections_limit = int(params["detections_limit"])
    ws = load_working_set(Path("working_set.yaml"))
    keys = [i.key for i in ws.all()]

    endpoint = os.environ["PLATFORM_API_ENDPOINT"]
    token = platform_api.get_access_token(
        endpoint, os.environ["PLATFORM_LOGIN"], os.environ["PLATFORM_PASSWORD"]
    )
    n = import_keys(
        store_dir=store_dir,
        keys=keys,
        api_endpoint=endpoint,
        token=token,
        detections_limit=detections_limit,
    )
    print(f"imported {n} sequences into {store_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run; expect pass**

Run: `uv run pytest tests/test_import_sequences.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/tube_builder_lab/platform_api.py src/tube_builder_lab/import_sequences.py scripts/import_sequences.py tests/test_import_sequences.py
git commit -m "feat(tube-lab): import working-set sequences by id"
```

---

## Task 9: Cache detections (run YOLO once)

**Files:**
- Create: `src/tube_builder_lab/cache.py`
- Create: `scripts/cache_detections.py`
- Create: `params.yaml`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test** (orchestration with an injected YOLO runner)

`tests/test_cache.py`:

```python
from pathlib import Path

from bbox_tube_temporal.types import Detection, FrameDetections
from pyrocore import Frame
from tube_builder_lab.cache import cache_one
from tube_builder_lab.detections_io import read_detections


def test_cache_one_writes_roundtrippable_detections(tmp_path: Path):
    frames = [
        Frame(frame_id="a", image_path=tmp_path / "a.jpg", timestamp=None),
        Frame(frame_id="b", image_path=tmp_path / "b.jpg", timestamp=None),
    ]
    canned = [
        FrameDetections(
            frame_idx=0, frame_id="a", timestamp=None,
            detections=[Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.8)],
        ),
        FrameDetections(frame_idx=1, frame_id="b", timestamp=None, detections=[]),
    ]
    out_dir = tmp_path / "detections"

    def fake_run_yolo(fs: list[Frame]) -> list[FrameDetections]:
        assert fs == frames
        return canned

    path = cache_one(out_dir=out_dir, key="platform_42", frames=frames, run_yolo=fake_run_yolo)
    assert path == out_dir / "platform_42.json"
    assert read_detections(path) == canned


def test_cache_one_skips_when_present(tmp_path: Path):
    out_dir = tmp_path / "detections"
    out_dir.mkdir()
    (out_dir / "platform_9.json").write_text("[]")
    calls = {"n": 0}

    def run_yolo(_):
        calls["n"] += 1
        return []

    cache_one(out_dir=out_dir, key="platform_9", frames=[], run_yolo=run_yolo, overwrite=False)
    assert calls["n"] == 0  # skipped, did not run YOLO
```

- [ ] **Step 2: Run; expect failure**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `cache.py`**

```python
"""Cache per-frame YOLO detections for the working-set sequences."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from bbox_tube_temporal.types import FrameDetections
from pyrocore import Frame

from .detections_io import read_detections, write_detections

log = logging.getLogger(__name__)

YoloRunner = Callable[[list[Frame]], list[FrameDetections]]


def cache_one(
    *,
    out_dir: Path,
    key: str,
    frames: list[Frame],
    run_yolo: YoloRunner,
    overwrite: bool = False,
) -> Path:
    """Run (or skip) YOLO for one sequence; write detections JSON; return path."""
    path = out_dir / f"{key}.json"
    if path.exists() and not overwrite:
        log.info("cache hit for %s; skipping", key)
        return path
    fds = run_yolo(frames)
    write_detections(path, fds)
    log.info("cached %d frames for %s", len(fds), key)
    return path


def detections_present(out_dir: Path, key: str) -> bool:
    return (out_dir / f"{key}.json").exists()


def load_cached(out_dir: Path, key: str) -> list[FrameDetections]:
    return read_detections(out_dir / f"{key}.json")
```

- [ ] **Step 4: Implement the CLI `scripts/cache_detections.py`**

```python
"""CLI: run the model's bundled YOLO over the working set, cache detections.

Loads a model package to reuse its EXACT YOLO weights + detection params, so
cached detections match BboxTubeTemporalModel.predict. Also writes the lab
pipeline config so the app stays model-free.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from bbox_tube_temporal.inference import run_yolo_on_frames
from bbox_tube_temporal.package import load_model_package

from tube_builder_lab.cache import cache_one
from tube_builder_lab.pipeline import extract_pipeline_config, write_pipeline_config
from tube_builder_lab.store import build_frames, read_meta, seq_dir_for_key
from tube_builder_lab.working_set import load_working_set

logging.basicConfig(level=logging.INFO)


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    store_dir = Path(params["store"])
    out_dir = Path(params["detections_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_zip = Path(params["models_dir"]) / params["model_name"] / "model.zip"

    pkg = load_model_package(model_zip)
    cfg = extract_pipeline_config(pkg.config)
    write_pipeline_config(Path(params["pipeline_config"]), cfg)

    def run_yolo(frames):
        return run_yolo_on_frames(
            pkg.yolo_model,
            frames,
            confidence_threshold=cfg.confidence_threshold,
            iou_nms=cfg.iou_nms,
            image_size=cfg.image_size,
        )

    ws = load_working_set(Path("working_set.yaml"))
    for item in ws.all():
        seq_dir = seq_dir_for_key(store_dir, item.key)
        if seq_dir is None:
            logging.warning("no sequence dir for %s; run import_sequences first", item.key)
            continue
        frames = build_frames(seq_dir, read_meta(seq_dir))
        cache_one(out_dir=out_dir, key=item.key, frames=frames, run_yolo=run_yolo)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `params.yaml`**

```yaml
# Lab configuration
model_name: vit_dinov2_finetune          # which data/06_models/<name>/model.zip to source YOLO + config from
detections_limit: 30                      # frames per sequence to import from the platform
store: data/03_primary/sequences
models_dir: data/06_models
detections_dir: data/05_model_input/detections
pipeline_config: data/05_model_input/pipeline_config.yaml
```

- [ ] **Step 6: Run; expect pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/tube_builder_lab/cache.py scripts/cache_detections.py params.yaml tests/test_cache.py
git commit -m "feat(tube-lab): cache YOLO detections for the working set"
```

---

## Task 10: The Streamlit app (Layout A)

**Files:**
- Create: `src/tube_builder_lab/app.py`

No unit tests (Streamlit UI); verified manually in Task 11. The app reads only
the cached data — it never loads a model or calls the API.

- [ ] **Step 1: Implement `app.py`**

```python
"""Tube Builder Lab — current vs candidate tube linking, Layout A.

Reads only data/05_model_input/detections/ + data/03_primary/sequences/**.
Run with `streamlit run app.py` (or `make app`). Iterate by editing
src/tube_builder_lab/candidate.py and clicking "Re-run candidate".
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Absolute imports: this module is the Streamlit entrypoint (run as __main__).
from tube_builder_lab import candidate as candidate_mod
from tube_builder_lab.cache import detections_present, load_cached
from tube_builder_lab.pipeline import (
    current_builder,
    detections_to_display_tubes,
    load_pipeline_config,
)
from tube_builder_lab.store import build_frames, read_meta, seq_dir_for_key
from tube_builder_lab.viz import (
    crop_tube_at_frame,
    draw_tube_bboxes,
    tube_color,
    tube_count,
    tube_timeline_df,
)
from tube_builder_lab.working_set import load_working_set

PARAMS = yaml.safe_load(Path("params.yaml").read_text())
STORE = Path(PARAMS["store"])
DETECTIONS = Path(PARAMS["detections_dir"])
PIPELINE_CONFIG = Path(PARAMS["pipeline_config"])
WORKING_SET = Path("working_set.yaml")
PLAY_FPS = 1


def _both_tube_sets(key: str, cfg, truncate: bool, _rev: int):
    """(current_tubes, candidate_tubes) for a key. _rev busts cache on Re-run."""
    fds = load_cached(DETECTIONS, key)
    cur = detections_to_display_tubes(fds, current_builder(cfg), cfg, truncate=truncate)
    cand = detections_to_display_tubes(
        fds, candidate_mod.build_tubes_candidate, cfg, truncate=truncate
    )
    return cur, cand


def _timeline_chart(alt, tubes, n, current, color_map):  # pragma: no cover
    df = tube_timeline_df(tubes)
    order = sorted(df["tube"].unique(), key=lambda t: int(t[1:])) if len(df) else []
    xscale = alt.Scale(domain=[0, n], nice=False)
    bars = (
        alt.Chart(df)
        .mark_bar(height=16, cornerRadius=3)
        .encode(
            x=alt.X("frame:Q", title="frame", scale=xscale,
                    axis=alt.Axis(format="d", tickMinStep=1)),
            x2="frame_end:Q",
            y=alt.Y("tube:N", title=None, sort=order),
            color=alt.Color("tube:N", sort=order,
                            scale=alt.Scale(domain=order, range=[color_map[o] for o in order]),
                            legend=None),
            opacity=alt.Opacity("is_gap:N",
                                scale=alt.Scale(domain=[False, True], range=[1.0, 0.4]),
                                legend=None),
            tooltip=[alt.Tooltip("tube:N"), alt.Tooltip("frame:Q"),
                     alt.Tooltip("confidence:Q", format=".2f"), alt.Tooltip("is_gap:N")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [current + 0.5]}))
        .mark_rule(color="#111", strokeDash=[4, 3], size=2)
        .encode(x=alt.X("x:Q", scale=xscale, title="frame"))
    )
    return alt.layer(bars, rule).properties(
        height=max(70, len(order) * 30),
        autosize={"type": "fit-x", "contains": "padding"},
    )


@st.fragment(run_every=1.0 / PLAY_FPS)
def _viewer(key: str, cfg, truncate: bool, rev: int):  # pragma: no cover
    import altair as alt  # noqa: PLC0415

    seq_dir = seq_dir_for_key(STORE, key)
    if seq_dir is None or not detections_present(DETECTIONS, key):
        st.warning(f"{key}: missing sequence frames or cached detections.")
        return
    meta = read_meta(seq_dir)
    frames = build_frames(seq_dir, meta)
    n = min(len(frames), cfg.max_frames) if truncate else len(frames)
    if not n:
        st.info("no frames")
        return

    cur, cand = _both_tube_sets(key, cfg, truncate, rev)
    c1, c2 = st.columns(2)
    c1.metric("current tubes", tube_count(cur))
    c2.metric("candidate tubes", tube_count(cand), delta=tube_count(cand) - tube_count(cur))

    color_by = st.radio("frame coloring", ["candidate", "current"], horizontal=True,
                        key=f"color_{key}")
    frame_key = f"frame_{key}"
    st.session_state.setdefault(frame_key, 0)
    if st.toggle("▶ play", value=True, key=f"play_{key}"):
        st.session_state[frame_key] = (st.session_state[frame_key] + 1) % n
    i = st.slider("frame", 0, n - 1, key=frame_key) if n > 1 else 0

    shown = cand if color_by == "candidate" else cur
    st.image(
        draw_tube_bboxes(seq_dir / meta.frames[i].file, shown, i),
        caption=f"frame {i + 1}/{n} — colored by {color_by}",
        width="stretch",
    )

    cur_colors = {f"T{t.tube_id}": tube_color(t.tube_id) for t in cur}
    cand_colors = {f"T{t.tube_id}": tube_color(t.tube_id) for t in cand}
    st.caption(f"current — {tube_count(cur)} tube(s)")
    if cur:
        st.altair_chart(_timeline_chart(alt, cur, n, i, cur_colors), width="stretch")
    st.caption(f"candidate — {tube_count(cand)} tube(s)")
    if cand:
        st.altair_chart(_timeline_chart(alt, cand, n, i, cand_colors), width="stretch")

    with st.expander("candidate crops @ this frame", expanded=False):
        cols = st.columns(max(1, len(cand)))
        for col, t in zip(cols, cand, strict=False):
            entry = next((e for e in t.entries if e.frame_idx == i and e.detection), None)
            col.markdown(f"**:{'red' if False else 'gray'}[T{t.tube_id}]**")
            if entry:
                d = entry.detection
                col.image(crop_tube_at_frame(seq_dir / meta.frames[i].file,
                                             (d.cx, d.cy, d.w, d.h)), width=180)
            else:
                col.caption("inactive")


def _summary(cfg, truncate: bool, rev: int) -> pd.DataFrame:  # pragma: no cover
    rows = []
    ws = load_working_set(WORKING_SET)
    for item in ws.all():
        if not detections_present(DETECTIONS, item.key):
            rows.append({"key": item.key, "current": None, "candidate": None,
                         "Δ": None, "note": item.note or ""})
            continue
        cur, cand = _both_tube_sets(item.key, cfg, truncate, rev)
        rows.append({"key": item.key, "current": len(cur), "candidate": len(cand),
                     "Δ": len(cand) - len(cur), "note": item.note or ""})
    return pd.DataFrame(rows)


def main() -> None:  # pragma: no cover
    st.set_page_config(page_title="Tube Builder Lab", layout="wide")
    st.title("Tube Builder Lab")

    if not PIPELINE_CONFIG.exists():
        st.warning("No pipeline config. Run `make cache` (cache_detections) first.")
        return
    cfg = load_pipeline_config(PIPELINE_CONFIG)
    ws = load_working_set(WORKING_SET)
    keys = [i.key for i in ws.all()]
    notes = {i.key: i.note for i in ws.all()}

    st.session_state.setdefault("rev", 0)
    with st.sidebar:
        st.header("Tube Lab")
        truncate = st.toggle("truncate to max_frames", value=True,
                             help=f"first {cfg.max_frames} frames (reproduces the model)")
        if st.button("🔄 Re-run candidate"):
            importlib.reload(candidate_mod)
            st.session_state["rev"] += 1
            st.toast("candidate.py reloaded")
        idx = st.selectbox("sequence", range(len(keys)),
                           format_func=lambda j: f"{keys[j]}  {('· ' + notes[keys[j]]) if notes[keys[j]] else ''}",
                           key="seq_idx")

    rev = st.session_state["rev"]
    key = keys[idx]
    if notes.get(key):
        st.caption(f"📝 {notes[key]}")
    _viewer(key, cfg, truncate, rev)

    st.divider()
    st.subheader("Working-set summary (current → candidate tube counts)")
    st.dataframe(_summary(cfg, truncate, rev), width="stretch", hide_index=True)


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 2: Lint + format**

Run: `make lint && make format`
Expected: no errors (format may reflow; re-run lint to confirm clean).

- [ ] **Step 3: Commit**

```bash
git add src/tube_builder_lab/app.py
git commit -m "feat(tube-lab): Layout A comparison app"
```

---

## Task 11: Operational bootstrap, DVC wiring, manual verification

**Files:**
- Create: `dvc.yaml` (cache_detections stage)
- Modify: `.gitignore` (ensure `data/` artifacts go through DVC, mirroring the template)

> Steps 1–3 are **operator-run** (need platform creds / GPU / DVC). Do not run
> `dvc pull`. The agent prepares files; the operator runs the imports/DVC.

- [ ] **Step 1 (operator): import the working-set sequences**

```bash
export PLATFORM_API_ENDPOINT=https://...
export PLATFORM_LOGIN=...
export PLATFORM_PASSWORD=...
uv run python scripts/import_sequences.py
```
Expected: `imported 16 sequences into data/03_primary/sequences`, each as
`data/03_primary/sequences/platform_<id>/{meta.json,images/*.jpg}`.

- [ ] **Step 2 (operator): provide a model package + cache detections**

Copy one model package in (any temporal variant — only its YOLO + config are
used), then cache:

```bash
mkdir -p data/06_models/vit_dinov2_finetune
cp ../temporal-model-explorer/data/06_models/vit_dinov2_finetune/model.zip data/06_models/vit_dinov2_finetune/
make cache
```
Expected: `data/05_model_input/detections/platform_<id>.json` for each key, plus
`data/05_model_input/pipeline_config.yaml`.

- [ ] **Step 3: Add the `dvc.yaml` cache stage** (reproducible detections)

```yaml
stages:
  cache_detections:
    cmd: uv run python scripts/cache_detections.py
    deps:
      - scripts/cache_detections.py
      - data/03_primary/sequences
      - data/06_models
      - working_set.yaml
      - params.yaml
    outs:
      - data/05_model_input/detections
      - data/05_model_input/pipeline_config.yaml
```

- [ ] **Step 4: Run the full test suite + lint**

Run: `make test && make lint`
Expected: all tests PASS; ruff clean.

- [ ] **Step 5 (operator): manual verification of the app**

Run: `make app` → open http://localhost:8501

Verify:
1. The sidebar lists all 16 sequences (notes shown next to keyed ones).
2. Selecting `platform_41304` shows the frame player, two stacked timelines
   (current above candidate), and `current N → candidate M` metrics.
3. With the candidate seeded as current, **current and candidate counts/timelines
   are identical** on every sequence and the summary `Δ` column is all `0`
   (sanity: the seed is a no-op).
4. The truncate toggle changes the frame count / timeline x-range.
5. Editing `src/tube_builder_lab/candidate.py` (e.g. set `MAX_MISSES = 5`) and
   clicking **🔄 Re-run candidate** updates the candidate timeline + summary
   without restarting.
6. The summary table renders one row per working-set key.

- [ ] **Step 6: Commit (code) and hand DVC push to the operator**

```bash
git add dvc.yaml .gitignore
git commit -m "build(tube-lab): dvc cache_detections stage + data tracking"
```
Then (operator): `uv run dvc add data/03_primary/sequences data/06_models` as
needed and `uv run dvc push` to publish. (Do not run `dvc pull`.)

---

## After the lab is built: iteration (the actual R&D)

The lab is now the durable artifact; the algorithm work happens by editing
`src/tube_builder_lab/candidate.py` against the working set:

1. Open a target sequence (start with the simplest, e.g. `platform_42538`
   "should be only one tube").
2. Read its current timeline to see *why* it fragments (adjacent split vs.
   distant re-detection).
3. Edit `candidate.py` (containment/IoMin association for adjacent splits; a
   post-hoc merge pass or larger `max_misses` for distant re-detections),
   click **Re-run candidate**, confirm the merge visually.
4. Add a behavior test to `tests/test_candidate.py` encoding the fix on
   synthetic detections.
5. Watch the summary `Δ` column: the control set (and untargeted sequences)
   must not newly over-merge.

Propagating the winning `build_tubes_candidate` into `lib/bbox-tube-temporal`
and retraining are a **separate PR** (out of scope here).

---

## Self-Review

**Spec coverage:**
- Full isolation / own DVC sequences → Tasks 1, 8, 11. ✓
- Detections via model's bundled YOLO, cached once → Task 9. ✓
- Editable, hot-reloaded candidate → Tasks 6, 10 (Re-run button). ✓
- Layout A (shared frame player + stacked timelines) → Task 10. ✓
- Truncation toggle, default ON, shared → Tasks 5, 10. ✓
- Working set (16 + notes) + control → Task 4; control left `[]` per spec. ✓
- Summary table for regressions → Task 10. ✓
- Viz adapted to Tube objects, pure helpers tested → Task 7. ✓
- Tests for io/pipeline/cache/import/viz/candidate → Tasks 2–9. ✓
- Out of scope (lib propagation, retrain) → stated in header + closing. ✓

**Placeholder scan:** no TBD/TODO; `control: []` is the spec-defined empty list,
not a placeholder; every code step shows complete code. ✓

**Type consistency:** `PipelineConfig` fields match between `extract_pipeline_config`,
`write/load_pipeline_config`, and the `cache_detections` CLI; `Builder` signature
(`list[FrameDetections] -> list[Tube]`) matches `current_builder`,
`build_tubes_candidate`, and `detections_to_display_tubes`; `params.yaml` keys
(`store`, `models_dir`, `model_name`, `detections_dir`, `pipeline_config`,
`detections_limit`) match every reader (`cache_detections.py`,
`import_sequences.py`, `app.py`). ✓

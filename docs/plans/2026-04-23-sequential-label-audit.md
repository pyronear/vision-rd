# Sequential Label Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold `experiments/data-quality/sequential/` — a self-contained uv+DVC experiment that runs a `pyrocore.TemporalModel` on every sequence of the pyro-dataset sequential splits (`train`/`val`/`test`), isolates disagreements against the folder-based ground truth, and surfaces them as FiftyOne FP/FN review sets.

**Architecture:** Standard template-derived experiment. Three `src/` modules (`registry`, `dataset`, `review`) split by responsibility; three thin CLI scripts (`predict.py`, `build_review_sets.py`, `build_fiftyone.py`) chained through a DVC pipeline that `foreach`es over splits per model. Cross-experiment inputs (dataset + packaged model) are `dvc import`ed with pinned revs; `.dvc` files are committed, teammates `dvc pull`.

**Tech Stack:** Python 3.11, uv, DVC (S3 remote), ruff, pytest, pyrocore, bbox-tube-temporal (as a uv path dependency), fiftyone (in an `explore` dep group).

## Spec

Design doc: `docs/specs/2026-04-23-sequential-label-audit-design.md` (commits `9022878` → `17abf7a` on `arthur/experiments-data-quality-check`).

## File layout built by this plan

```
experiments/data-quality/sequential/
├── .dvc/config                        # S3 remote for this experiment
├── .gitignore                         # venv, caches, raw data
├── .python-version                    # "3.11"
├── Makefile                           # install/lint/format/test
├── README.md                          # purpose, reproduce steps
├── data/
│   ├── 01_raw/                        # dvc-imported inputs
│   │   ├── datasets/{train,val,test}.dvc
│   │   └── models/bbox-tube-temporal-vit-dinov2-finetune.zip.dvc
│   ├── 07_model_output/               # predict stage output (gitignored, DVC-tracked)
│   ├── 08_reporting/                  # review-set stage output (DVC-tracked)
│   └── fiftyone/                      # build_fiftyone stage sentinel (DVC-tracked)
├── dvc.yaml                           # predict → build_review_sets → build_fiftyone
├── notebooks/                         # empty, for ad-hoc exploration
├── params.yaml                        # splits list + models map
├── pyproject.toml                     # uv project + path deps
├── scripts/
│   ├── predict.py                     # run one (model, split)
│   ├── build_review_sets.py           # extract FP/FN + write CSV + manifest
│   └── build_fiftyone.py              # build FP/FN FiftyOne datasets from manifest
├── src/data_quality_sequential/
│   ├── __init__.py
│   ├── registry.py                    # MODEL_REGISTRY + load_model()
│   ├── dataset.py                     # SequenceRef + iter_sequences()
│   └── review.py                      # Prediction + ReviewSet + build_review_sets()
├── tests/
│   ├── __init__.py
│   ├── test_registry.py
│   ├── test_dataset.py
│   └── test_review.py
└── uv.lock
```

Module responsibilities:

| File | Responsibility |
|---|---|
| `src/.../registry.py` | Short-name → model class table + factory via `.from_package(path)` |
| `src/.../dataset.py` | Scan a split directory; emit `SequenceRef(name, split, ground_truth, frame_paths)` |
| `src/.../review.py` | Given `SequenceRef` list + `Prediction` list, produce FP and FN `ReviewSet`s |
| `scripts/predict.py` | Thin CLI: load model from package, iterate sequences, dump `predictions.json` + `results.json` |
| `scripts/build_review_sets.py` | Thin CLI: read predictions + dataset, write CSVs + summary + manifest |
| `scripts/build_fiftyone.py` | Thin CLI: read manifest, create/refresh two FiftyOne datasets, write sentinel JSON |

---

## Task 1: Scaffold the experiment directory from the template

**Files:**
- Create: `experiments/data-quality/sequential/` (via copy of `experiments/template/`)
- Create: `experiments/data-quality/sequential/src/data_quality_sequential/__init__.py`
- Modify: `experiments/data-quality/sequential/pyproject.toml`
- Modify: `experiments/data-quality/sequential/.dvc/config`
- Modify: `experiments/data-quality/sequential/README.md` (replace with stub; final content in Task 12)

- [ ] **Step 1: Copy the template**

Run (from repo root):

```bash
mkdir -p experiments/data-quality
cp -r experiments/template experiments/data-quality/sequential
```

- [ ] **Step 2: Rename the package directory**

```bash
cd experiments/data-quality/sequential
mv src/project_name src/data_quality_sequential
```

- [ ] **Step 3: Confirm skeleton files exist**

```bash
ls -a
ls src/data_quality_sequential
```

Expected: `.dvc .gitignore .python-version Makefile README.md configs data dvc.yaml notebooks pyproject.toml scripts src tests`; inside `src/data_quality_sequential/` expect an `__init__.py` (if absent, create an empty file with `printf '' > src/data_quality_sequential/__init__.py`).

- [ ] **Step 4: Rewrite `pyproject.toml`**

Replace the entire file contents with:

```toml
[project]
name = "data-quality-sequential"
version = "0.1.0"
description = "Use TemporalModel oracles to surface mis-labeled sequences in the pyro-dataset"
requires-python = ">=3.11"
dependencies = [
    "pyrocore",
    "bbox-tube-temporal",
    "tqdm>=4.67.3",
]

[tool.uv.sources]
pyrocore = { path = "../../../lib/pyrocore" }
bbox-tube-temporal = { path = "../../temporal-models/bbox-tube-temporal" }

[dependency-groups]
dev = [
    "dvc[s3]>=3.56",
    "nbqa>=1.9",
    "nbstripout>=0.8",
    "pytest>=8.0",
    "ruff>=0.9",
]
explore = [
    "fiftyone>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/data_quality_sequential"]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 5: Point the DVC remote at this experiment**

Rewrite `.dvc/config`:

```ini
[core]
    remote = s3remote
    analytics = false
['remote "s3remote"']
    url = s3://pyro-vision-rd/dvc/experiments/data-quality/sequential/
```

- [ ] **Step 6: Replace README with a placeholder**

Replace `README.md` with:

```markdown
# data-quality/sequential

Surfaces probably-mis-labeled sequences in the pyro-dataset by running a
`pyrocore.TemporalModel` oracle over every sequence and presenting
disagreements against the folder-based ground truth as FiftyOne FP/FN
review sets.

See `../../../docs/specs/2026-04-23-sequential-label-audit-design.md` for design.
Reproduce steps are filled in at the end of the implementation.
```

- [ ] **Step 7: Install**

```bash
uv sync
```

Expected: creates `.venv/`, writes `uv.lock`. No errors.

- [ ] **Step 8: Commit**

```bash
git add experiments/data-quality/sequential/.dvc/config \
        experiments/data-quality/sequential/.gitignore \
        experiments/data-quality/sequential/.python-version \
        experiments/data-quality/sequential/Makefile \
        experiments/data-quality/sequential/README.md \
        experiments/data-quality/sequential/configs \
        experiments/data-quality/sequential/data \
        experiments/data-quality/sequential/dvc.yaml \
        experiments/data-quality/sequential/notebooks \
        experiments/data-quality/sequential/pyproject.toml \
        experiments/data-quality/sequential/scripts \
        experiments/data-quality/sequential/src \
        experiments/data-quality/sequential/tests \
        experiments/data-quality/sequential/uv.lock
git commit -m "chore(data-quality/sequential): scaffold experiment from template"
```

---

## Task 2: Implement `dataset.py` (TDD)

**Files:**
- Create: `experiments/data-quality/sequential/tests/test_dataset.py`
- Create: `experiments/data-quality/sequential/src/data_quality_sequential/dataset.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset.py`:

```python
"""Tests for data_quality_sequential.dataset."""

from pathlib import Path

from data_quality_sequential.dataset import SequenceRef, iter_sequences


def _make_split(
    tmp_path: Path, split: str, wildfire: list[str], fp: list[str]
) -> Path:
    split_dir = tmp_path / split
    for name in wildfire:
        seq = split_dir / "wildfire" / name / "images"
        seq.mkdir(parents=True)
        (seq / f"{name}_2023-05-23T17-18-31.jpg").touch()
        (seq / f"{name}_2023-05-23T17-18-01.jpg").touch()
    for name in fp:
        seq = split_dir / "fp" / name / "images"
        seq.mkdir(parents=True)
        (seq / f"{name}_2023-05-23T18-00-00.jpg").touch()
    return split_dir


def test_iter_sequences_emits_ground_truth_from_parent_dir(tmp_path: Path) -> None:
    split_dir = _make_split(
        tmp_path, "val", wildfire=["wf_a", "wf_b"], fp=["fp_a"]
    )

    refs = list(iter_sequences(split_dir, split="val"))

    by_name = {r.name: r for r in refs}
    assert set(by_name) == {"wf_a", "wf_b", "fp_a"}
    assert by_name["wf_a"].ground_truth is True
    assert by_name["wf_b"].ground_truth is True
    assert by_name["fp_a"].ground_truth is False
    assert all(r.split == "val" for r in refs)


def test_iter_sequences_returns_frames_sorted_by_filename(tmp_path: Path) -> None:
    split_dir = _make_split(tmp_path, "val", wildfire=["wf_a"], fp=[])

    [ref] = list(iter_sequences(split_dir, split="val"))

    assert len(ref.frame_paths) == 2
    # Filename-sort puts 17-18-01 before 17-18-31.
    assert ref.frame_paths[0].name.endswith("17-18-01.jpg")
    assert ref.frame_paths[1].name.endswith("17-18-31.jpg")


def test_iter_sequences_skips_sequences_with_no_images(tmp_path: Path) -> None:
    split_dir = tmp_path / "val"
    empty = split_dir / "wildfire" / "empty_seq" / "images"
    empty.mkdir(parents=True)
    populated = split_dir / "wildfire" / "ok_seq" / "images"
    populated.mkdir(parents=True)
    (populated / "ok_seq_2023-05-23T17-18-01.jpg").touch()

    refs = list(iter_sequences(split_dir, split="val"))
    names = {r.name for r in refs}

    assert names == {"ok_seq"}


def test_iter_sequences_handles_missing_wildfire_or_fp_dir(tmp_path: Path) -> None:
    split_dir = _make_split(tmp_path, "test", wildfire=[], fp=["fp_only"])

    refs = list(iter_sequences(split_dir, split="test"))

    assert [r.name for r in refs] == ["fp_only"]
    assert refs[0].ground_truth is False


def test_sequence_ref_is_a_dataclass_with_expected_fields() -> None:
    ref = SequenceRef(
        name="x",
        split="val",
        ground_truth=True,
        frame_paths=[Path("/tmp/x_2023-01-01T00-00-00.jpg")],
    )
    assert ref.name == "x"
    assert ref.split == "val"
    assert ref.ground_truth is True
    assert ref.frame_paths[0].name.endswith(".jpg")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'data_quality_sequential.dataset'`.

- [ ] **Step 3: Implement `dataset.py`**

Create `src/data_quality_sequential/dataset.py`:

```python
"""Sequence discovery + folder-based ground truth.

Scans a pyro-dataset sequential split (``<split>/{wildfire,fp}/<seq>/images/*.jpg``)
and emits :class:`SequenceRef` records with ground truth inferred from the parent
directory (``wildfire/`` = positive, ``fp/`` = negative).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SequenceRef:
    """One sequence ready to be fed to a :class:`pyrocore.TemporalModel`.

    Attributes:
        name: Sequence directory name (unique within a split).
        split: ``"train"`` | ``"val"`` | ``"test"``.
        ground_truth: ``True`` iff the sequence lives under ``wildfire/``.
        frame_paths: Frame image paths, sorted by filename.
    """

    name: str
    split: str
    ground_truth: bool
    frame_paths: list[Path]


def _collect(group_dir: Path, split: str, ground_truth: bool) -> list[SequenceRef]:
    if not group_dir.is_dir():
        return []
    refs: list[SequenceRef] = []
    for seq_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        images_dir = seq_dir / "images"
        if not images_dir.is_dir():
            continue
        frames = sorted(images_dir.glob("*.jpg"))
        if not frames:
            continue
        refs.append(
            SequenceRef(
                name=seq_dir.name,
                split=split,
                ground_truth=ground_truth,
                frame_paths=frames,
            )
        )
    return refs


def iter_sequences(split_dir: Path, split: str) -> Iterator[SequenceRef]:
    """Yield every sequence in ``split_dir`` with its ground-truth label.

    Sequences with no ``images/`` directory or no ``*.jpg`` files are skipped.

    Args:
        split_dir: Root of a single split (e.g., ``data/01_raw/datasets/train``).
        split: Label attached to each emitted :class:`SequenceRef` (typically
            the directory name).

    Yields:
        :class:`SequenceRef` for each non-empty sequence under ``wildfire/`` and
        then ``fp/``.
    """
    yield from _collect(split_dir / "wildfire", split=split, ground_truth=True)
    yield from _collect(split_dir / "fp", split=split, ground_truth=False)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint and format**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

Expected: no errors; `ruff format` reports files reformatted or already formatted.

- [ ] **Step 6: Commit**

```bash
git add src/data_quality_sequential/dataset.py tests/test_dataset.py
git commit -m "feat(data-quality/sequential): dataset scanner with folder-based ground truth"
```

---

## Task 3: Implement `registry.py` (TDD)

**Files:**
- Create: `experiments/data-quality/sequential/tests/test_registry.py`
- Create: `experiments/data-quality/sequential/src/data_quality_sequential/registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry.py`:

```python
"""Tests for data_quality_sequential.registry."""

from pathlib import Path

import pytest

from data_quality_sequential.registry import MODEL_REGISTRY, load_model


def test_registry_exposes_bbox_tube_temporal() -> None:
    assert "bbox-tube-temporal" in MODEL_REGISTRY
    module_path, class_name = MODEL_REGISTRY["bbox-tube-temporal"]
    assert module_path == "bbox_tube_temporal.model"
    assert class_name == "BboxTubeTemporalModel"


def test_load_model_raises_on_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model type"):
        load_model("does-not-exist", tmp_path / "missing.zip")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'data_quality_sequential.registry'`.

- [ ] **Step 3: Implement `registry.py`**

Create `src/data_quality_sequential/registry.py`:

```python
"""Registry of :class:`pyrocore.TemporalModel` implementations addressable by short name.

Single source of truth for the ``--model-type`` flag shared by all scripts.
Each entry is a ``(module_path, class_name)`` pair for a class that exposes
``classmethod from_package(path) -> Self``.
"""

import importlib
from pathlib import Path

from pyrocore import TemporalModel

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "bbox-tube-temporal": (
        "bbox_tube_temporal.model",
        "BboxTubeTemporalModel",
    ),
}


def load_model(model_type: str, package_path: Path) -> TemporalModel:
    """Instantiate a :class:`TemporalModel` from *package_path*.

    Args:
        model_type: Registry key (must be in :data:`MODEL_REGISTRY`).
        package_path: Path to the packaged ``.zip`` produced by the model's
            own packaging stage.

    Raises:
        ValueError: If ``model_type`` is not a registered key.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {model_type!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    module_path, class_name = MODEL_REGISTRY[model_type]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls.from_package(package_path)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Lint and format**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

- [ ] **Step 6: Commit**

```bash
git add src/data_quality_sequential/registry.py tests/test_registry.py
git commit -m "feat(data-quality/sequential): model registry with bbox-tube-temporal entry"
```

---

## Task 4: Implement `review.py` (TDD)

**Files:**
- Create: `experiments/data-quality/sequential/tests/test_review.py`
- Create: `experiments/data-quality/sequential/src/data_quality_sequential/review.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_review.py`:

```python
"""Tests for data_quality_sequential.review."""

from pathlib import Path

from data_quality_sequential.dataset import SequenceRef
from data_quality_sequential.review import (
    Prediction,
    ReviewEntry,
    ReviewSet,
    build_review_sets,
)


def _ref(name: str, ground_truth: bool) -> SequenceRef:
    return SequenceRef(
        name=name,
        split="val",
        ground_truth=ground_truth,
        frame_paths=[Path(f"/tmp/{name}_2023-01-01T00-00-00.jpg")],
    )


def _pred(name: str, predicted: bool, trigger: int | None = None) -> Prediction:
    return Prediction(
        sequence_name=name,
        predicted=predicted,
        trigger_frame_index=trigger,
    )


def test_build_review_sets_partitions_fp_and_fn() -> None:
    refs = [
        _ref("wf_a", True),
        _ref("wf_b", True),
        _ref("fp_a", False),
        _ref("fp_b", False),
    ]
    preds = [
        _pred("wf_a", predicted=True, trigger=3),    # TP
        _pred("wf_b", predicted=False),              # FN
        _pred("fp_a", predicted=True, trigger=0),    # FP
        _pred("fp_b", predicted=False),              # TN
    ]

    fp, fn = build_review_sets(refs, preds, split="val", model_name="m")

    assert isinstance(fp, ReviewSet) and isinstance(fn, ReviewSet)
    assert fp.kind == "fp"
    assert fn.kind == "fn"
    assert [e.sequence_name for e in fp.entries] == ["fp_a"]
    assert [e.sequence_name for e in fn.entries] == ["wf_b"]


def test_review_entries_are_sorted_alphabetically() -> None:
    refs = [
        _ref("wf_z", True),
        _ref("wf_a", True),
        _ref("fp_z", False),
        _ref("fp_a", False),
    ]
    preds = [
        _pred("wf_z", False),
        _pred("wf_a", False),
        _pred("fp_z", True, trigger=5),
        _pred("fp_a", True, trigger=0),
    ]

    fp, fn = build_review_sets(refs, preds, split="val", model_name="m")

    assert [e.sequence_name for e in fp.entries] == ["fp_a", "fp_z"]
    assert [e.sequence_name for e in fn.entries] == ["wf_a", "wf_z"]


def test_review_entries_carry_model_and_split_metadata() -> None:
    refs = [_ref("fp_a", False)]
    preds = [_pred("fp_a", True, trigger=2)]

    fp, _ = build_review_sets(refs, preds, split="test", model_name="my-model")

    [entry] = fp.entries
    assert entry == ReviewEntry(
        sequence_name="fp_a",
        split="test",
        model_name="my-model",
        ground_truth=False,
        predicted=True,
        trigger_frame_index=2,
    )


def test_missing_predictions_are_skipped_not_fatal() -> None:
    refs = [_ref("wf_a", True), _ref("fp_a", False)]
    preds = [_pred("wf_a", False)]  # fp_a has no prediction (e.g., model errored)

    fp, fn = build_review_sets(refs, preds, split="val", model_name="m")

    # wf_a is FN; fp_a simply doesn't appear.
    assert [e.sequence_name for e in fn.entries] == ["wf_a"]
    assert fp.entries == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_review.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'data_quality_sequential.review'`.

- [ ] **Step 3: Implement `review.py`**

Create `src/data_quality_sequential/review.py`:

```python
"""FP / FN extraction from a model's per-sequence predictions.

Given a list of :class:`~data_quality_sequential.dataset.SequenceRef` ground
truths and a list of :class:`Prediction`s, splits disagreements into two
:class:`ReviewSet`s:

* FP — predicted positive, ground truth negative.
* FN — predicted negative, ground truth positive.

Both sets are emitted unranked in stable alphabetical order by sequence name.
Ranking is deliberately deferred (see
``docs/specs/2026-04-23-sequential-label-audit-design.md`` §8).
"""

from dataclasses import dataclass
from typing import Literal

from .dataset import SequenceRef


@dataclass
class Prediction:
    """One model's verdict on a single sequence."""

    sequence_name: str
    predicted: bool
    trigger_frame_index: int | None = None


@dataclass
class ReviewEntry:
    """One sequence flagged for human review."""

    sequence_name: str
    split: str
    model_name: str
    ground_truth: bool
    predicted: bool
    trigger_frame_index: int | None


@dataclass
class ReviewSet:
    """All :class:`ReviewEntry`s of one kind for one (model, split)."""

    kind: Literal["fp", "fn"]
    split: str
    model_name: str
    entries: list[ReviewEntry]


def build_review_sets(
    refs: list[SequenceRef],
    predictions: list[Prediction],
    *,
    split: str,
    model_name: str,
) -> tuple[ReviewSet, ReviewSet]:
    """Return ``(fp_set, fn_set)`` for one ``(model, split)``.

    Sequences without a matching :class:`Prediction` are silently skipped.
    Within each set, entries are sorted alphabetically by ``sequence_name``.
    """
    pred_by_name = {p.sequence_name: p for p in predictions}

    fp_entries: list[ReviewEntry] = []
    fn_entries: list[ReviewEntry] = []

    for ref in refs:
        pred = pred_by_name.get(ref.name)
        if pred is None:
            continue
        entry = ReviewEntry(
            sequence_name=ref.name,
            split=split,
            model_name=model_name,
            ground_truth=ref.ground_truth,
            predicted=pred.predicted,
            trigger_frame_index=pred.trigger_frame_index,
        )
        if pred.predicted and not ref.ground_truth:
            fp_entries.append(entry)
        elif not pred.predicted and ref.ground_truth:
            fn_entries.append(entry)

    fp_entries.sort(key=lambda e: e.sequence_name)
    fn_entries.sort(key=lambda e: e.sequence_name)

    return (
        ReviewSet(kind="fp", split=split, model_name=model_name, entries=fp_entries),
        ReviewSet(kind="fn", split=split, model_name=model_name, entries=fn_entries),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_review.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint and format**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

- [ ] **Step 6: Commit**

```bash
git add src/data_quality_sequential/review.py tests/test_review.py
git commit -m "feat(data-quality/sequential): build_review_sets splits FP/FN"
```

---

## Task 5: Script — `predict.py`

**Files:**
- Create: `experiments/data-quality/sequential/scripts/predict.py`

- [ ] **Step 1: Write `scripts/predict.py`**

Create `scripts/predict.py`:

```python
"""Run a TemporalModel on every sequence of one split, dump predictions.

Usage::

    uv run python scripts/predict.py \
        --model-name bbox-tube-temporal-vit-dinov2-finetune \
        --model-type bbox-tube-temporal \
        --model-package data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip \
        --split-dir data/01_raw/datasets/val \
        --split val \
        --output-dir data/07_model_output/bbox-tube-temporal-vit-dinov2-finetune/val
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from tqdm import tqdm

from data_quality_sequential.dataset import iter_sequences
from data_quality_sequential.registry import MODEL_REGISTRY, load_model
from data_quality_sequential.review import Prediction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--model-type", required=True, choices=sorted(MODEL_REGISTRY)
    )
    parser.add_argument("--model-package", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s from %s", args.model_type, args.model_package)
    model = load_model(args.model_type, args.model_package)

    refs = list(iter_sequences(args.split_dir, split=args.split))
    logger.info("Found %d sequences in %s", len(refs), args.split_dir)

    predictions: list[Prediction] = []
    for ref in tqdm(refs, desc=f"predict[{args.split}]", unit="seq"):
        output = model.predict_sequence(ref.frame_paths)
        predictions.append(
            Prediction(
                sequence_name=ref.name,
                predicted=output.is_positive,
                trigger_frame_index=output.trigger_frame_index,
            )
        )

    preds_path = args.output_dir / "predictions.json"
    preds_path.write_text(
        json.dumps([dataclasses.asdict(p) for p in predictions], indent=2)
    )
    logger.info("Wrote %d predictions to %s", len(predictions), preds_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--help` renders without crashing**

```bash
uv run python scripts/predict.py --help
```

Expected: argparse help text listing the six flags. No traceback.

- [ ] **Step 3: Lint and format**

```bash
uv run ruff check scripts/
uv run ruff format scripts/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/predict.py
git commit -m "feat(data-quality/sequential): predict.py runs one model on one split"
```

---

## Task 6: Script — `build_review_sets.py`

**Files:**
- Create: `experiments/data-quality/sequential/scripts/build_review_sets.py`

- [ ] **Step 1: Write `scripts/build_review_sets.py`**

Create `scripts/build_review_sets.py`:

```python
"""Turn per-sequence predictions + folder ground truth into FP/FN review sets.

Outputs (per (model, split)):
  - ``fp_sequences.csv``         — one row per FP, ordered by sequence name.
  - ``fn_sequences.csv``         — one row per FN, ordered by sequence name.
  - ``summary.json``              — counts + confusion matrix.
  - ``review_manifest.json``      — machine-readable input for build_fiftyone.

Usage::

    uv run python scripts/build_review_sets.py \
        --model-name bbox-tube-temporal-vit-dinov2-finetune \
        --split val \
        --split-dir data/01_raw/datasets/val \
        --predictions-path data/07_model_output/bbox-tube-temporal-vit-dinov2-finetune/val/predictions.json \
        --output-dir data/08_reporting/bbox-tube-temporal-vit-dinov2-finetune/val
"""

import argparse
import csv
import dataclasses
import json
import logging
from pathlib import Path

from data_quality_sequential.dataset import iter_sequences
from data_quality_sequential.review import (
    Prediction,
    ReviewSet,
    build_review_sets,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "sequence_name",
    "split",
    "model_name",
    "ground_truth",
    "predicted",
    "trigger_frame_index",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _load_predictions(path: Path) -> list[Prediction]:
    raw = json.loads(path.read_text())
    return [Prediction(**entry) for entry in raw]


def _write_csv(review_set: ReviewSet, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for entry in review_set.entries:
            writer.writerow(dataclasses.asdict(entry))


def _confusion_matrix(refs, preds_by_name) -> dict:
    tp = fp = fn = tn = missing = 0
    for ref in refs:
        pred = preds_by_name.get(ref.name)
        if pred is None:
            missing += 1
            continue
        if ref.ground_truth and pred.predicted:
            tp += 1
        elif not ref.ground_truth and pred.predicted:
            fp += 1
        elif ref.ground_truth and not pred.predicted:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "missing": missing}


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    refs = list(iter_sequences(args.split_dir, split=args.split))
    predictions = _load_predictions(args.predictions_path)
    logger.info(
        "Loaded %d refs and %d predictions for %s/%s",
        len(refs),
        len(predictions),
        args.model_name,
        args.split,
    )

    fp_set, fn_set = build_review_sets(
        refs, predictions, split=args.split, model_name=args.model_name
    )

    _write_csv(fp_set, args.output_dir / "fp_sequences.csv")
    _write_csv(fn_set, args.output_dir / "fn_sequences.csv")

    summary = {
        "model_name": args.model_name,
        "split": args.split,
        "num_refs": len(refs),
        "num_predictions": len(predictions),
        "confusion_matrix": _confusion_matrix(
            refs, {p.sequence_name: p for p in predictions}
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    manifest = {
        "model_name": args.model_name,
        "split": args.split,
        "split_dir": str(args.split_dir),
        "fp": [dataclasses.asdict(e) for e in fp_set.entries],
        "fn": [dataclasses.asdict(e) for e in fn_set.entries],
    }
    (args.output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    logger.info(
        "Wrote FP=%d FN=%d to %s", len(fp_set.entries), len(fn_set.entries),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--help`**

```bash
uv run python scripts/build_review_sets.py --help
```

Expected: argparse help text. No traceback.

- [ ] **Step 3: Lint and format**

```bash
uv run ruff check scripts/
uv run ruff format scripts/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_review_sets.py
git commit -m "feat(data-quality/sequential): build_review_sets.py emits CSV + manifest"
```

---

## Task 7: Script — `build_fiftyone.py`

**Files:**
- Create: `experiments/data-quality/sequential/scripts/build_fiftyone.py`

- [ ] **Step 1: Write `scripts/build_fiftyone.py`**

Create `scripts/build_fiftyone.py`:

```python
"""Build FiftyOne FP and FN review datasets from a review manifest.

One FiftyOne dataset per error kind, per (model, split). Each sample is one
frame; samples are grouped by a ``sequence_name`` field so reviewers can
browse sequence by sequence.

Usage::

    uv run --group explore python scripts/build_fiftyone.py \
        --manifest-path data/08_reporting/<model>/<split>/review_manifest.json \
        --output-dir data/fiftyone/<model>/<split>
"""

import argparse
import json
import logging
from pathlib import Path

import fiftyone as fo

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _frames_for_sequence(split_dir: Path, ground_truth: bool, name: str) -> list[Path]:
    bucket = "wildfire" if ground_truth else "fp"
    images_dir = split_dir / bucket / name / "images"
    if not images_dir.is_dir():
        return []
    return sorted(images_dir.glob("*.jpg"))


def _build_dataset(
    dataset_name: str,
    split_dir: Path,
    entries: list[dict],
) -> int:
    """Create (or overwrite) a FiftyOne dataset; return number of samples added."""
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name, persistent=True)

    samples: list[fo.Sample] = []
    for entry in entries:
        frames = _frames_for_sequence(
            split_dir=split_dir,
            ground_truth=entry["ground_truth"],
            name=entry["sequence_name"],
        )
        for frame_idx, frame_path in enumerate(frames):
            samples.append(
                fo.Sample(
                    filepath=str(frame_path),
                    sequence_name=entry["sequence_name"],
                    split=entry["split"],
                    model_name=entry["model_name"],
                    ground_truth=entry["ground_truth"],
                    predicted=entry["predicted"],
                    trigger_frame_index=entry["trigger_frame_index"],
                    frame_index=frame_idx,
                )
            )
    if samples:
        dataset.add_samples(samples)
    return len(samples)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest_path.read_text())
    model_name = manifest["model_name"]
    split = manifest["split"]
    split_dir = Path(manifest["split_dir"])

    fp_dataset_name = f"dq-seq_{model_name}_{split}_fp"
    fn_dataset_name = f"dq-seq_{model_name}_{split}_fn"

    fp_samples = _build_dataset(fp_dataset_name, split_dir, manifest["fp"])
    fn_samples = _build_dataset(fn_dataset_name, split_dir, manifest["fn"])

    sentinel = {
        "model_name": model_name,
        "split": split,
        "fp_dataset": fp_dataset_name,
        "fp_sequences": len(manifest["fp"]),
        "fp_samples": fp_samples,
        "fn_dataset": fn_dataset_name,
        "fn_sequences": len(manifest["fn"]),
        "fn_samples": fn_samples,
    }
    (args.output_dir / "datasets.json").write_text(json.dumps(sentinel, indent=2))

    logger.info(
        "Built FiftyOne datasets %s (%d samples) and %s (%d samples)",
        fp_dataset_name,
        fp_samples,
        fn_dataset_name,
        fn_samples,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `--help`**

```bash
uv run --group explore python scripts/build_fiftyone.py --help
```

Expected: argparse help text. (First run downloads fiftyone — that's fine.)

- [ ] **Step 3: Lint and format**

```bash
uv run ruff check scripts/
uv run ruff format scripts/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_fiftyone.py
git commit -m "feat(data-quality/sequential): build_fiftyone.py creates FP/FN review datasets"
```

---

## Task 8: Run all unit tests

**Files:** none — validation step.

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: 11 passed (5 dataset + 2 registry + 4 review). No failures, no warnings about missing modules.

- [ ] **Step 2: Run ruff across everything**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: `All checks passed!` for check, and no files need reformatting.

- [ ] **Step 3: Commit (only if earlier steps left staged changes; otherwise skip)**

If `git status` shows nothing, skip this step.

---

## Task 9: DVC-import the sequential dataset

Pulls `train`, `val`, and `test` splits from `pyronear/pyro-dataset`. User preference: do **not** run `dvc pull`; teammates do that manually after cloning.

**Files:**
- Create: `experiments/data-quality/sequential/data/01_raw/datasets/train.dvc`
- Create: `experiments/data-quality/sequential/data/01_raw/datasets/val.dvc`
- Create: `experiments/data-quality/sequential/data/01_raw/datasets/test.dvc`

- [ ] **Step 1: Pick the pyro-dataset tag**

Find the tag currently in use by other experiments in this repo:

```bash
cat ../../temporal-models/temporal-model-leaderboard/data/01_raw/sequential_test.dvc
```

Look at the `repo.rev` / `rev` field. As of the most recent commit on `main`, the canonical tag is **`v3.0.0`** (see `experiments/GUIDELINES.md` and recent commit `87a9820 chore(experiments): regenerate metrics on pyro-dataset v3.0.0`). Use `v3.0.0` unless a newer tag is clearly the current default. If uncertain, stop and ask.

Store the chosen tag in a shell variable for the next commands:

```bash
PYRO_DATASET_TAG=v3.0.0
```

- [ ] **Step 2: Import train**

```bash
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/train \
    -o data/01_raw/datasets/train \
    --rev "$PYRO_DATASET_TAG"
```

Expected: creates `data/01_raw/datasets/train.dvc`, prints "Importing..." then success. **This also downloads the data into `data/01_raw/datasets/train/` — that's fine, it's how `dvc import` works.**

- [ ] **Step 3: Import val**

```bash
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/val \
    -o data/01_raw/datasets/val \
    --rev "$PYRO_DATASET_TAG"
```

Expected: creates `data/01_raw/datasets/val.dvc`.

- [ ] **Step 4: Import test**

```bash
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_test \
    -o data/01_raw/datasets/test \
    --rev "$PYRO_DATASET_TAG"
```

Expected: creates `data/01_raw/datasets/test.dvc`.

- [ ] **Step 5: Sanity-check the layout**

```bash
ls data/01_raw/datasets/
ls data/01_raw/datasets/train/
ls data/01_raw/datasets/train/wildfire/ | head -3
ls data/01_raw/datasets/train/fp/ | head -3
```

Expected: `.dvc` files for each split; `wildfire/` and `fp/` subdirs with many sequence dirs.

- [ ] **Step 6: Commit the three `.dvc` files**

```bash
git add data/01_raw/datasets/train.dvc \
        data/01_raw/datasets/val.dvc \
        data/01_raw/datasets/test.dvc
git commit -m "data(data-quality/sequential): import pyro-dataset v3.0.0 splits via dvc"
```

Note: the `train/`, `val/`, `test/` directories themselves should be `.gitignore`d via the template's default `.gitignore` (data directories are generally gitignored in these experiments — confirm with `git status`; if the raw directories appear as untracked, add them to a local `.gitignore` pattern like `data/01_raw/datasets/*/` — but the `.dvc` files must remain tracked).

---

## Task 10: DVC-import the bbox-tube-temporal model package

**Files:**
- Create: `experiments/data-quality/sequential/data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip.dvc`

- [ ] **Step 1: Pick a git ref of this repo with the packaged model pushed**

The source is `experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip` on **this repo**. For `dvc import` to succeed, the chosen ref must have been `dvc push`ed for that experiment.

Check which refs have the zip tracked:

```bash
cd ../../..  # back to repo root
git log --oneline --all --follow \
    experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip.dvc \
    2>/dev/null | head
git log --oneline main -- \
    experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip \
    2>/dev/null | head
cd experiments/data-quality/sequential
```

Pick a ref that is both (a) present in both `origin/main` and (b) has had `dvc push` run. The safest default is the tip of `main` (the leaderboard already imports this variant, so it has been pushed there at least).

Store the chosen ref:

```bash
MODEL_REF=main   # replace with a tag/commit if you need a pinned version
```

**If it is unclear whether the zip has been pushed at the chosen ref, stop and ask.** Do NOT run `dvc push` from this task — the prerequisite lives in the `bbox-tube-temporal` experiment.

- [ ] **Step 2: Create the models directory**

```bash
mkdir -p data/01_raw/models
```

- [ ] **Step 3: `dvc import` the model zip**

Run from `experiments/data-quality/sequential/`. The repo URL is the remote `origin` URL of this repo:

```bash
REPO_URL=$(git -C ../../.. config --get remote.origin.url)
uv run dvc import "$REPO_URL" \
    experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip \
    -o data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip \
    --rev "$MODEL_REF"
```

Expected: creates `data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip.dvc` and downloads the zip into `data/01_raw/models/`.

- [ ] **Step 4: Verify the zip is usable**

```bash
ls -lh data/01_raw/models/
```

Expected: `bbox-tube-temporal-vit-dinov2-finetune.zip` present, ~150 MB.

- [ ] **Step 5: Commit the `.dvc` file**

```bash
git add data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip.dvc
git commit -m "data(data-quality/sequential): import bbox-tube-temporal dinov2 finetune zip"
```

---

## Task 11: Wire the pipeline — `params.yaml` + `dvc.yaml`

Starts with one model (`bbox-tube-temporal-vit-dinov2-finetune`). Adding more models later is a matter of (a) dvc-importing another zip, (b) adding an entry in `params.yaml` `models:`, (c) copy-pasting the three stage blocks with the new name.

**Files:**
- Create: `experiments/data-quality/sequential/params.yaml`
- Modify: `experiments/data-quality/sequential/dvc.yaml`

- [ ] **Step 1: Write `params.yaml`**

Replace the file contents with:

```yaml
splits:
  - train
  - val
  - test

models:
  bbox-tube-temporal-vit-dinov2-finetune:
    model_type: bbox-tube-temporal
```

- [ ] **Step 2: Write `dvc.yaml`**

Replace the file contents with (three stages; `foreach` expands over splits; adding a model means appending another triple of stage blocks):

```yaml
stages:
  # ---- bbox-tube-temporal-vit-dinov2-finetune ----

  predict_bbox_tube_temporal_vit_dinov2_finetune:
    foreach:
      - train
      - val
      - test
    do:
      desc: "Run bbox-tube-temporal (dinov2 finetune) on ${item}"
      cmd: >-
        uv run python scripts/predict.py
        --model-name bbox-tube-temporal-vit-dinov2-finetune
        --model-type bbox-tube-temporal
        --model-package data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip
        --split-dir data/01_raw/datasets/${item}
        --split ${item}
        --output-dir data/07_model_output/bbox-tube-temporal-vit-dinov2-finetune/${item}
      deps:
        - scripts/predict.py
        - src/data_quality_sequential
        - data/01_raw/datasets/${item}
        - data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip
      outs:
        - data/07_model_output/bbox-tube-temporal-vit-dinov2-finetune/${item}

  build_review_sets_bbox_tube_temporal_vit_dinov2_finetune:
    foreach:
      - train
      - val
      - test
    do:
      desc: "Extract FP/FN review sets for bbox-tube-temporal (dinov2 finetune) on ${item}"
      cmd: >-
        uv run python scripts/build_review_sets.py
        --model-name bbox-tube-temporal-vit-dinov2-finetune
        --split ${item}
        --split-dir data/01_raw/datasets/${item}
        --predictions-path data/07_model_output/bbox-tube-temporal-vit-dinov2-finetune/${item}/predictions.json
        --output-dir data/08_reporting/bbox-tube-temporal-vit-dinov2-finetune/${item}
      deps:
        - scripts/build_review_sets.py
        - src/data_quality_sequential
        - data/01_raw/datasets/${item}
        - data/07_model_output/bbox-tube-temporal-vit-dinov2-finetune/${item}/predictions.json
      outs:
        - data/08_reporting/bbox-tube-temporal-vit-dinov2-finetune/${item}

  build_fiftyone_bbox_tube_temporal_vit_dinov2_finetune:
    foreach:
      - train
      - val
      - test
    do:
      desc: "Build FiftyOne FP/FN datasets for bbox-tube-temporal (dinov2 finetune) on ${item}"
      cmd: >-
        uv run --group explore python scripts/build_fiftyone.py
        --manifest-path data/08_reporting/bbox-tube-temporal-vit-dinov2-finetune/${item}/review_manifest.json
        --output-dir data/fiftyone/bbox-tube-temporal-vit-dinov2-finetune/${item}
      deps:
        - scripts/build_fiftyone.py
        - data/08_reporting/bbox-tube-temporal-vit-dinov2-finetune/${item}/review_manifest.json
      outs:
        - data/fiftyone/bbox-tube-temporal-vit-dinov2-finetune/${item}
```

- [ ] **Step 3: Validate the DVC graph**

```bash
uv run dvc status
uv run dvc dag
```

Expected: `dvc status` reports all stages "changed" (since none have been run yet). `dvc dag` prints an ASCII DAG with 9 stages (3 splits × 3 stages) plus the imported inputs.

- [ ] **Step 4: Dry-run repro on val only**

```bash
uv run dvc repro -s predict_bbox_tube_temporal_vit_dinov2_finetune@val --dry
```

Expected: prints the command that would run for the `val` predict stage. No errors.

- [ ] **Step 5: Commit `params.yaml` + `dvc.yaml`**

```bash
git add params.yaml dvc.yaml
git commit -m "feat(data-quality/sequential): params + dvc pipeline for dinov2 finetune"
```

---

## Task 12: End-to-end repro on the `val` split

This is the first real execution. Running on `val` alone is a cheap smoke (a few hundred sequences) before committing to train + test.

**Files:**
- Modify/create: `experiments/data-quality/sequential/dvc.lock`
- Create (DVC-tracked, not git-tracked): `data/07_model_output/…/val/predictions.json`, `data/08_reporting/…/val/{fp,fn}_sequences.csv + summary.json + review_manifest.json`, `data/fiftyone/…/val/datasets.json`

- [ ] **Step 1: Run predict on val**

```bash
uv run dvc repro predict_bbox_tube_temporal_vit_dinov2_finetune@val
```

Expected: GPU or CPU inference runs; a tqdm bar counts `val` sequences; on completion, `data/07_model_output/bbox-tube-temporal-vit-dinov2-finetune/val/predictions.json` exists and contains one entry per sequence.

- [ ] **Step 2: Run build_review_sets on val**

```bash
uv run dvc repro build_review_sets_bbox_tube_temporal_vit_dinov2_finetune@val
```

Expected: `data/08_reporting/bbox-tube-temporal-vit-dinov2-finetune/val/` contains `fp_sequences.csv`, `fn_sequences.csv`, `summary.json`, `review_manifest.json`.

- [ ] **Step 3: Inspect `summary.json`**

```bash
cat data/08_reporting/bbox-tube-temporal-vit-dinov2-finetune/val/summary.json
```

Expected: JSON with `confusion_matrix` {tp, fp, fn, tn, missing}. `tp + fp + fn + tn + missing` should equal `num_refs`. Sanity-check: `fp + tn` ≈ number of `val/fp/` sequences, `tp + fn` ≈ number of `val/wildfire/` sequences.

- [ ] **Step 4: Run build_fiftyone on val**

```bash
uv run dvc repro build_fiftyone_bbox_tube_temporal_vit_dinov2_finetune@val
```

Expected: `data/fiftyone/bbox-tube-temporal-vit-dinov2-finetune/val/datasets.json` exists; its contents name `dq-seq_bbox-tube-temporal-vit-dinov2-finetune_val_fp` and `..._val_fn` FiftyOne datasets.

- [ ] **Step 5: Spot-check FiftyOne datasets**

```bash
uv run --group explore python -c '
import fiftyone as fo
for d in sorted(fo.list_datasets()):
    if d.startswith("dq-seq_"):
        ds = fo.load_dataset(d)
        print(f"{d}: {len(ds)} samples, fields={list(ds.get_field_schema())}")
'
```

Expected: two datasets listed with sample counts matching `summary.json`'s FP/FN sequence counts × (average sequence length). `fields` includes `sequence_name`, `split`, `model_name`, `ground_truth`, `predicted`, `trigger_frame_index`, `frame_index`.

- [ ] **Step 6: Commit `dvc.lock`**

```bash
git add dvc.lock
git commit -m "chore(data-quality/sequential): dvc.lock after val smoke run"
```

---

## Task 13: Full `dvc repro` on train + test

- [ ] **Step 1: Run the full pipeline**

```bash
uv run dvc repro
```

Expected: `val` stages are up-to-date (skipped); `train` and `test` stages run through predict → build_review_sets → build_fiftyone. Total runtime depends on hardware but runs unattended.

- [ ] **Step 2: Verify all outputs exist**

```bash
find data/07_model_output data/08_reporting data/fiftyone -type f | sort
```

Expected: for each of train/val/test:
- `data/07_model_output/<model>/<split>/predictions.json`
- `data/08_reporting/<model>/<split>/{fp,fn}_sequences.csv`
- `data/08_reporting/<model>/<split>/{summary,review_manifest}.json`
- `data/fiftyone/<model>/<split>/datasets.json`

- [ ] **Step 3: Commit the updated `dvc.lock`**

```bash
git add dvc.lock
git commit -m "chore(data-quality/sequential): full dvc repro (train + val + test)"
```

- [ ] **Step 4: Push DVC outputs**

`dvc push` uploads pipeline artifacts so teammates can `dvc pull`. Only run this after confirming the outputs look right.

```bash
uv run dvc push
```

Expected: uploads new objects to `s3://pyro-vision-rd/dvc/experiments/data-quality/sequential/`. No errors.

---

## Task 14: Finalize the README

**Files:**
- Modify: `experiments/data-quality/sequential/README.md`

- [ ] **Step 1: Rewrite `README.md`**

Replace the placeholder with:

```markdown
# data-quality/sequential

Use a `pyrocore.TemporalModel` oracle to surface probably-mis-labeled
sequences in the pyro-dataset sequential splits (`train`/`val`/`test`).
For each registered model, the pipeline runs inference on every sequence
and flags disagreements against folder-based ground truth
(`wildfire/` = positive, `fp/` = negative) as two FiftyOne review sets:

- **FP set**: model predicted positive, sequence is under `fp/`.
- **FN set**: model predicted negative, sequence is under `wildfire/`.

Reviewers browse the FiftyOne datasets and decide whether each flag is
a real label error or a true model mistake.

## Design

See [`../../../docs/specs/2026-04-23-sequential-label-audit-design.md`](../../../docs/specs/2026-04-23-sequential-label-audit-design.md).

## Pipeline

```
predict  →  build_review_sets  →  build_fiftyone
 (split × model)     (split × model)     (split × model)
```

Adding a new model variant: see "Adding another model" below.

## How to reproduce

```bash
cd experiments/data-quality/sequential
make install

# Fetch the imported dataset + model zip:
uv run dvc pull

# Full pipeline (train + val + test × all models):
uv run dvc repro
```

Reports land in `data/08_reporting/<model>/<split>/`; FiftyOne datasets
named `dq-seq_<model>_<split>_fp` and `dq-seq_<model>_<split>_fn` are
created in the local FiftyOne mongo store.

Browse a review set:

```bash
uv run --group explore python -c '
import fiftyone as fo
session = fo.launch_app(fo.load_dataset("dq-seq_bbox-tube-temporal-vit-dinov2-finetune_val_fp"))
session.wait()
'
```

## Data imports

`train`/`val`/`test` are imported from [`pyro-dataset`](https://github.com/pyronear/pyro-dataset)
at tag `v3.0.0`:

```bash
uv run dvc import https://github.com/pyronear/pyro-dataset \
    data/processed/sequential_train_val/train \
    -o data/01_raw/datasets/train --rev v3.0.0
# (and val, test)
```

Model packages are imported from this repo (the `bbox-tube-temporal`
experiment must have `dvc push`ed the chosen variant at the pinned ref
first):

```bash
uv run dvc import <this-repo-url> \
    experiments/temporal-models/bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip \
    -o data/01_raw/models/bbox-tube-temporal-vit-dinov2-finetune.zip \
    --rev <tag-or-branch>
```

## Adding another model

1. `dvc import` its packaged `.zip` into `data/01_raw/models/<model-name>.zip`
   (filename must equal the model-name key used below).
2. Add to `params.yaml`:

   ```yaml
   models:
     <model-name>:
       model_type: <registry-key-from-src/data_quality_sequential/registry.py>
   ```

3. If `<registry-key>` isn't already in `MODEL_REGISTRY`, add it (one line)
   and verify the model class has a `classmethod from_package`.
4. Append three new stage blocks to `dvc.yaml` (copy the existing triple,
   replace the model-name everywhere).
5. `uv run dvc repro` — only the new model's stages run.

## Layout

```
data/
  01_raw/
    datasets/{train,val,test}.dvc       # dvc-imported
    models/<model>.zip.dvc              # dvc-imported
  07_model_output/<model>/<split>/
    predictions.json
  08_reporting/<model>/<split>/
    fp_sequences.csv
    fn_sequences.csv
    summary.json
    review_manifest.json
  fiftyone/<model>/<split>/
    datasets.json                       # sentinel; actual datasets in mongo
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(data-quality/sequential): README with pipeline + add-a-model guide"
```

---

## Task 15: Verify CI discovers the new experiment

The repo's experiments CI auto-discovers sub-projects under `experiments/**`. This task verifies nothing about the new experiment breaks linting or tests.

**Files:** none — validation step.

- [ ] **Step 1: Lint and format check**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 2: Full test run**

```bash
uv run pytest tests/ -v
```

Expected: 11 passed.

- [ ] **Step 3: DVC status**

```bash
uv run dvc status
```

Expected: `Data and pipelines are up to date.` (If anything is "changed", investigate — likely a stale output.)

- [ ] **Step 4: Inspect final git log**

```bash
git log --oneline arthur/experiments-data-quality-check ^main | head -20
```

Expected: one commit per task (roughly), all titled `feat/chore/data/docs(data-quality/sequential): …`.

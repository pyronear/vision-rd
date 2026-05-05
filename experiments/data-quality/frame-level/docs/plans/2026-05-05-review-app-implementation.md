# Frame-level review app — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-hosted review app inside `experiments/data-quality/frame-level/` that lets reviewers edit GT bboxes inline against YOLO predictions, auto-persist to `review.json`, and export corrected YOLO labels for `pyro-dataset`.

**Architecture:** FastAPI backend serves predictions + GT + review state; single-page vanilla JS frontend with HTML5 canvas bbox editor; auto-save with atomic write; live recomputation of TP/FP/FN matching when threshold sliders change.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pytest, vanilla JS + HTML5 canvas (no build step). Reuses existing `BBox` and `PredBBox` dataclasses from `data_quality_frame_level.dataset` and `.inference`.

**Spec:** [`2026-05-05-review-app-design.md`](../specs/2026-05-05-review-app-design.md)

---

## File structure

All paths are relative to `experiments/data-quality/frame-level/`.

| File | Responsibility |
|---|---|
| `src/data_quality_frame_level/review_app/__init__.py` | Package marker |
| `src/data_quality_frame_level/review_app/sequence.py` | `parse_stem` (split on last `_`); group + sort helpers |
| `src/data_quality_frame_level/review_app/matching.py` | `iou`, `evaluate_frame` returning per-bbox TP/FP/FN tags |
| `src/data_quality_frame_level/review_app/persistence.py` | `SampleReview`, `ReviewState` dataclasses; atomic JSON read/write |
| `src/data_quality_frame_level/review_app/queue.py` | `build_queue` from state + view + thresholds, with sequence sort key |
| `src/data_quality_frame_level/review_app/state.py` | `AppState` — load predictions + GT + review for a `(model, split)` |
| `src/data_quality_frame_level/review_app/export.py` | `compute_diff`, `export_corrections` (only-changed YOLO + manifest) |
| `src/data_quality_frame_level/review_app/main.py` | FastAPI app factory + routes |
| `src/data_quality_frame_level/review_app/static/index.html` | Workbench page scaffold |
| `src/data_quality_frame_level/review_app/static/app.css` | Workbench styling |
| `src/data_quality_frame_level/review_app/static/app.js` | All client-side behavior |
| `scripts/run_review_app.py` | uvicorn entrypoint with `--port` |
| `scripts/export_review_app.py` | CLI wrapper around `export.export_corrections` |
| `tests/test_sequence.py` | sequence.py unit tests |
| `tests/test_matching.py` | matching.py unit tests |
| `tests/test_persistence.py` | persistence.py unit tests (round-trip + atomicity) |
| `tests/test_queue.py` | queue.py unit tests |
| `tests/test_export.py` | export.py unit tests |
| `tests/test_main.py` | FastAPI route tests via TestClient |
| `Makefile` | `review-app`, `review-export` targets |
| `pyproject.toml` | Adds `review-app` dep group |

---

## Task 1: Scaffolding — deps, package, test dir

**Files:**
- Modify: `experiments/data-quality/frame-level/pyproject.toml`
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/__init__.py`

- [ ] **Step 1: Add `review-app` dependency group**

In `pyproject.toml`, under `[dependency-groups]`, add:

```toml
review-app = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "httpx>=0.27",  # for FastAPI TestClient in tests
]
```

- [ ] **Step 2: Create the package directory**

```bash
mkdir -p experiments/data-quality/frame-level/src/data_quality_frame_level/review_app
touch experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/__init__.py
```

- [ ] **Step 3: Sync deps**

```bash
cd experiments/data-quality/frame-level && uv sync --group review-app
```

Expected: lockfile updates, no errors.

- [ ] **Step 4: Verify import**

```bash
cd experiments/data-quality/frame-level && uv run python -c "import data_quality_frame_level.review_app"
```

Expected: no output (clean import).

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/pyproject.toml experiments/data-quality/frame-level/uv.lock experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/__init__.py
git commit -m "feat(data-quality/frame-level): scaffold review_app package + deps"
```

---

## Task 2: `sequence.py` — stem parsing + sort key

**Files:**
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/sequence.py`
- Create: `experiments/data-quality/frame-level/tests/test_review_app_sequence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_review_app_sequence.py
from data_quality_frame_level.review_app.sequence import parse_stem


def test_parse_stem_pyronear():
    assert parse_stem("pyronear-force-06_courmettes_275_2024-02-17T17-36-57") == (
        "pyronear-force-06_courmettes_275",
        "2024-02-17T17-36-57",
    )


def test_parse_stem_hyphenated_source():
    assert parse_stem("awf-axis_baldca_999_2023-06-04T07-35-26") == (
        "awf-axis_baldca_999",
        "2023-06-04T07-35-26",
    )


def test_parse_stem_no_hyphen_in_source():
    assert parse_stem("adf_avinyonet_999_2023-05-23T17-21-00") == (
        "adf_avinyonet_999",
        "2023-05-23T17-21-00",
    )
```

- [ ] **Step 2: Run tests; confirm they fail**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_sequence.py -v
```

Expected: `ModuleNotFoundError: No module named 'data_quality_frame_level.review_app.sequence'`.

- [ ] **Step 3: Implement `sequence.py`**

```python
# src/data_quality_frame_level/review_app/sequence.py
"""Stem parsing and sequence grouping for the review app.

Stems in pyro-dataset are
``<source>_<camera>_<sequence_id>_<timestamp>`` where ``<timestamp>``
is ISO-8601 with hyphen-replaced colons (e.g. ``2024-02-17T17-36-57``).
``<source>`` may contain hyphens (``awf-axis``, ``pyronear-force-06``)
but never underscores. Splitting on the last ``_`` reliably yields
``(sequence_id, timestamp)``.
"""


def parse_stem(stem: str) -> tuple[str, str]:
    """Return ``(sequence_id, timestamp)`` for a pyro-dataset stem."""
    sequence_id, timestamp = stem.rsplit("_", 1)
    return sequence_id, timestamp
```

- [ ] **Step 4: Run tests; confirm pass**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_sequence.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/sequence.py experiments/data-quality/frame-level/tests/test_review_app_sequence.py
git commit -m "feat(data-quality/frame-level): review_app sequence parser"
```

---

## Task 3: `matching.py` — IoU + per-frame TP/FP/FN

**Files:**
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/matching.py`
- Create: `experiments/data-quality/frame-level/tests/test_review_app_matching.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_review_app_matching.py
from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.inference import PredBBox
from data_quality_frame_level.review_app.matching import (
    EvaluatedFrame,
    evaluate_frame,
    iou,
)


def _gt(cx, cy, w, h):
    return BBox(class_id=0, cx=cx, cy=cy, w=w, h=h)


def _pred(cx, cy, w, h, conf):
    return PredBBox(class_id=0, cx=cx, cy=cy, w=w, h=h, conf=conf)


def test_iou_identical():
    b = _gt(0.5, 0.5, 0.2, 0.2)
    assert iou(b, b) == 1.0


def test_iou_disjoint():
    a = _gt(0.1, 0.1, 0.1, 0.1)
    b = _gt(0.9, 0.9, 0.1, 0.1)
    assert iou(a, b) == 0.0


def test_evaluate_frame_tp_fp_fn():
    gt = [_gt(0.5, 0.5, 0.2, 0.2), _gt(0.8, 0.8, 0.1, 0.1)]
    preds = [
        _pred(0.5, 0.5, 0.2, 0.2, 0.9),  # matches gt[0] -> TP
        _pred(0.1, 0.1, 0.1, 0.1, 0.6),  # unmatched   -> FP
    ]
    out = evaluate_frame(gt=gt, predictions=preds, iou_thresh=0.5)
    assert isinstance(out, EvaluatedFrame)
    assert out.gt_status == ["tp", "fn"]
    assert out.pred_status == ["tp", "fp"]


def test_evaluate_frame_iou_threshold_filters():
    gt = [_gt(0.5, 0.5, 0.2, 0.2)]
    preds = [_pred(0.6, 0.6, 0.2, 0.2, 0.9)]  # low IoU pair
    strict = evaluate_frame(gt=gt, predictions=preds, iou_thresh=0.9)
    lenient = evaluate_frame(gt=gt, predictions=preds, iou_thresh=0.05)
    assert strict.gt_status == ["fn"] and strict.pred_status == ["fp"]
    assert lenient.gt_status == ["tp"] and lenient.pred_status == ["tp"]
```

- [ ] **Step 2: Run; confirm fail**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_matching.py -v
```

Expected: import error.

- [ ] **Step 3: Implement `matching.py`**

```python
# src/data_quality_frame_level/review_app/matching.py
"""Per-frame TP / FP / FN assignment using greedy IoU matching.

Mirrors FiftyOne's ``evaluate_detections`` for our single-class case so
the app is independent of FiftyOne. Predictions and GT are matched
greedily by descending IoU; unmatched predictions become FP, unmatched
GT becomes FN.
"""

from dataclasses import dataclass

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.inference import PredBBox


@dataclass(frozen=True)
class EvaluatedFrame:
    gt_status: list[str]      # one per GT bbox: "tp" | "fn"
    pred_status: list[str]    # one per prediction: "tp" | "fp"
    matches: list[tuple[int, int, float]]  # (gt_idx, pred_idx, iou)


def iou(a: BBox | PredBBox, b: BBox | PredBBox) -> float:
    ax1, ay1 = a.cx - a.w / 2, a.cy - a.h / 2
    ax2, ay2 = a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1 = b.cx - b.w / 2, b.cy - b.h / 2
    bx2, by2 = b.cx + b.w / 2, b.cy + b.h / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def evaluate_frame(
    *,
    gt: list[BBox],
    predictions: list[PredBBox],
    iou_thresh: float,
) -> EvaluatedFrame:
    candidates = sorted(
        (
            (i, j, iou(g, p))
            for i, g in enumerate(gt)
            for j, p in enumerate(predictions)
        ),
        key=lambda x: x[2],
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gi, pj, score in candidates:
        if score < iou_thresh:
            break
        if gi in matched_gt or pj in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pj)
        matches.append((gi, pj, score))
    gt_status = ["tp" if i in matched_gt else "fn" for i in range(len(gt))]
    pred_status = [
        "tp" if j in matched_pred else "fp" for j in range(len(predictions))
    ]
    return EvaluatedFrame(
        gt_status=gt_status, pred_status=pred_status, matches=matches
    )
```

- [ ] **Step 4: Run; confirm pass**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_matching.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/matching.py experiments/data-quality/frame-level/tests/test_review_app_matching.py
git commit -m "feat(data-quality/frame-level): review_app TP/FP/FN matching"
```

---

## Task 4: `persistence.py` — review.json schema + atomic IO

**Files:**
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/persistence.py`
- Create: `experiments/data-quality/frame-level/tests/test_review_app_persistence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_review_app_persistence.py
import json
from pathlib import Path

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.persistence import (
    ReviewState,
    SampleReview,
    read_review_state,
    write_review_state,
)


def _bb(cx, cy, w=0.1, h=0.1):
    return BBox(class_id=0, cx=cx, cy=cy, w=w, h=h)


def test_read_missing_file_returns_empty(tmp_path: Path):
    state = read_review_state(tmp_path / "review.json", model="m", split="val")
    assert state.samples == {}
    assert state.model == "m"
    assert state.split == "val"


def test_round_trip(tmp_path: Path):
    p = tmp_path / "review.json"
    state = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(
                status="reviewed",
                bboxes=[_bb(0.5, 0.5)],
                reviewer="arthur",
                note="ok",
                reviewed_at="2026-05-05T14:00:00Z",
            )
        },
    )
    write_review_state(p, state)
    payload = json.loads(p.read_text())
    assert payload["version"] == 1
    assert payload["model_name"] == "m"
    assert payload["split"] == "val"
    assert "stem_a" in payload["samples"]
    reloaded = read_review_state(p, model="m", split="val")
    assert reloaded == state


def test_write_is_atomic(tmp_path: Path):
    p = tmp_path / "review.json"
    write_review_state(p, ReviewState(model="m", split="val", samples={}))
    assert p.exists()
    assert not (tmp_path / "review.json.tmp").exists()


def test_serialization_is_sorted(tmp_path: Path):
    p = tmp_path / "review.json"
    state = ReviewState(
        model="m",
        split="val",
        samples={
            "z": SampleReview(status="reviewed", bboxes=[]),
            "a": SampleReview(status="reviewed", bboxes=[]),
        },
    )
    write_review_state(p, state)
    text = p.read_text()
    assert text.index('"a"') < text.index('"z"')
```

- [ ] **Step 2: Run; confirm fail**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_persistence.py -v
```

Expected: import error.

- [ ] **Step 3: Implement `persistence.py`**

```python
# src/data_quality_frame_level/review_app/persistence.py
"""Atomic read/write of ``review.json`` per ``(model, split)``.

The file shape is documented in the design spec §5.1. Writes go through
a sibling ``.tmp`` + ``os.replace`` so partial writes can never be
observed. Reads of missing files return an empty :class:`ReviewState`.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from data_quality_frame_level.dataset import BBox

PAYLOAD_VERSION = 1
ALLOWED_STATUS = ("reviewed", "unclear")


@dataclass
class SampleReview:
    status: str
    bboxes: list[BBox] = field(default_factory=list)
    reviewer: str | None = None
    note: str | None = None
    reviewed_at: str | None = None


@dataclass
class ReviewState:
    model: str
    split: str
    samples: dict[str, SampleReview] = field(default_factory=dict)


def _bbox_to_dict(b: BBox) -> dict:
    return {"class_id": b.class_id, "cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h}


def _dict_to_bbox(d: dict) -> BBox:
    return BBox(
        class_id=int(d["class_id"]),
        cx=float(d["cx"]),
        cy=float(d["cy"]),
        w=float(d["w"]),
        h=float(d["h"]),
    )


def _sample_to_dict(s: SampleReview) -> dict:
    out: dict = {
        "status": s.status,
        "bboxes": [_bbox_to_dict(b) for b in s.bboxes],
    }
    if s.reviewer is not None:
        out["reviewer"] = s.reviewer
    if s.note is not None:
        out["note"] = s.note
    if s.reviewed_at is not None:
        out["reviewed_at"] = s.reviewed_at
    return out


def _dict_to_sample(d: dict) -> SampleReview:
    if d["status"] not in ALLOWED_STATUS:
        raise ValueError(f"unknown status: {d['status']!r}")
    return SampleReview(
        status=d["status"],
        bboxes=[_dict_to_bbox(b) for b in d.get("bboxes", [])],
        reviewer=d.get("reviewer"),
        note=d.get("note"),
        reviewed_at=d.get("reviewed_at"),
    )


def read_review_state(path: Path, *, model: str, split: str) -> ReviewState:
    if not path.is_file():
        return ReviewState(model=model, split=split)
    payload = json.loads(path.read_text())
    if payload.get("version") != PAYLOAD_VERSION:
        raise ValueError(f"unsupported review.json version: {payload.get('version')}")
    return ReviewState(
        model=payload.get("model_name", model),
        split=payload.get("split", split),
        samples={
            stem: _dict_to_sample(d)
            for stem, d in sorted(payload.get("samples", {}).items())
        },
    )


def write_review_state(path: Path, state: ReviewState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PAYLOAD_VERSION,
        "model_name": state.model,
        "split": state.split,
        "samples": {
            stem: _sample_to_dict(state.samples[stem])
            for stem in sorted(state.samples)
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
```

- [ ] **Step 4: Run; confirm pass**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_persistence.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/persistence.py experiments/data-quality/frame-level/tests/test_review_app_persistence.py
git commit -m "feat(data-quality/frame-level): review_app persistence with atomic write"
```

---

## Task 5: `queue.py` — view-aware queue building

**Files:**
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/queue.py`
- Create: `experiments/data-quality/frame-level/tests/test_review_app_queue.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_review_app_queue.py
from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.inference import PredBBox
from data_quality_frame_level.review_app.queue import QueueItem, build_queue


def _gt(cx=0.5, cy=0.5, w=0.1, h=0.1):
    return BBox(class_id=0, cx=cx, cy=cy, w=w, h=h)


def _pred(conf, cx=0.5, cy=0.5, w=0.1, h=0.1):
    return PredBBox(class_id=0, cx=cx, cy=cy, w=w, h=h, conf=conf)


def test_fp_queue_groups_sequences_by_max_confidence():
    # seqA has a 0.9 FP, seqB has a 0.6 FP → seqA first; within seqA,
    # frames are in timestamp order.
    predictions = {
        "seqA_2024-01-01T00-00-00": [_pred(0.7)],
        "seqA_2024-01-01T00-00-30": [_pred(0.9)],
        "seqB_2024-01-01T00-00-00": [_pred(0.6)],
    }
    gt: dict[str, list[BBox]] = {k: [] for k in predictions}
    queue = build_queue(
        predictions=predictions,
        gt=gt,
        review_status={},
        view="fp",
        conf_thresh=0.05,
        iou_thresh=0.05,
        review_conf_thresh=0.5,
    )
    stems = [item.stem for item in queue]
    assert stems == [
        "seqA_2024-01-01T00-00-00",
        "seqA_2024-01-01T00-00-30",
        "seqB_2024-01-01T00-00-00",
    ]
    assert all(isinstance(item, QueueItem) for item in queue)


def test_fp_queue_filters_by_review_conf():
    predictions = {"s_t": [_pred(0.4)]}
    gt: dict[str, list[BBox]] = {"s_t": []}
    out = build_queue(
        predictions=predictions,
        gt=gt,
        review_status={},
        view="fp",
        conf_thresh=0.05,
        iou_thresh=0.05,
        review_conf_thresh=0.5,
    )
    assert out == []


def test_fn_queue_sorts_by_max_gt_area():
    predictions: dict[str, list[PredBBox]] = {
        "seqA_2024-01-01T00-00-00": [],
        "seqB_2024-01-01T00-00-00": [],
    }
    gt = {
        "seqA_2024-01-01T00-00-00": [_gt(w=0.1, h=0.1)],
        "seqB_2024-01-01T00-00-00": [_gt(w=0.3, h=0.3)],
    }
    out = build_queue(
        predictions=predictions,
        gt=gt,
        review_status={},
        view="fn",
        conf_thresh=0.05,
        iou_thresh=0.05,
        review_conf_thresh=0.0,
    )
    assert [i.stem for i in out] == [
        "seqB_2024-01-01T00-00-00",
        "seqA_2024-01-01T00-00-00",
    ]
```

- [ ] **Step 2: Run; confirm fail**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_queue.py -v
```

Expected: import error.

- [ ] **Step 3: Implement `queue.py`**

```python
# src/data_quality_frame_level/review_app/queue.py
"""Queue building for the review app.

Filters and sorts the universe of `(stem, predictions, gt)` triples
into a list of :class:`QueueItem` for the active view. Sort order
clusters siblings within a sequence (so adjacent items are also
adjacent in time), with sequences ordered by their max severity for
the active view (FP confidence, FN area, or the max of the two for
``all``).
"""

from dataclasses import dataclass

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.inference import PredBBox
from data_quality_frame_level.review_app.matching import evaluate_frame
from data_quality_frame_level.review_app.sequence import parse_stem


@dataclass(frozen=True)
class QueueItem:
    stem: str
    sequence_id: str
    timestamp: str
    kind: str           # "fp" | "fn" | "mixed"
    severity: float     # FP conf or FN area depending on view
    status: str | None  # "reviewed" | "unclear" | None


def _frame_severity(
    *,
    gt: list[BBox],
    predictions: list[PredBBox],
    conf_thresh: float,
    iou_thresh: float,
    review_conf_thresh: float,
    view: str,
) -> tuple[str | None, float]:
    pred_filt = [p for p in predictions if p.conf >= conf_thresh]
    ev = evaluate_frame(gt=gt, predictions=pred_filt, iou_thresh=iou_thresh)
    fp_conf = max(
        (p.conf for p, s in zip(pred_filt, ev.pred_status, strict=True)
         if s == "fp" and p.conf >= review_conf_thresh),
        default=0.0,
    )
    fn_area = max(
        (g.w * g.h for g, s in zip(gt, ev.gt_status, strict=True) if s == "fn"),
        default=0.0,
    )
    if view == "fp" and fp_conf > 0.0:
        return "fp", fp_conf
    if view == "fn" and fn_area > 0.0:
        return "fn", fn_area
    if view == "all":
        if fp_conf > 0.0 or fn_area > 0.0:
            return ("mixed" if fp_conf > 0.0 and fn_area > 0.0
                    else ("fp" if fp_conf > 0.0 else "fn")), max(fp_conf, fn_area)
    return None, 0.0


def build_queue(
    *,
    predictions: dict[str, list[PredBBox]],
    gt: dict[str, list[BBox]],
    review_status: dict[str, str],
    view: str,
    conf_thresh: float,
    iou_thresh: float,
    review_conf_thresh: float,
) -> list[QueueItem]:
    items: list[QueueItem] = []
    for stem in sorted(predictions.keys() | gt.keys()):
        kind, severity = _frame_severity(
            gt=gt.get(stem, []),
            predictions=predictions.get(stem, []),
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            review_conf_thresh=review_conf_thresh,
            view=view,
        )
        if kind is None:
            continue
        seq_id, ts = parse_stem(stem)
        items.append(
            QueueItem(
                stem=stem,
                sequence_id=seq_id,
                timestamp=ts,
                kind=kind,
                severity=severity,
                status=review_status.get(stem),
            )
        )

    seq_max: dict[str, float] = {}
    for it in items:
        if it.severity > seq_max.get(it.sequence_id, 0.0):
            seq_max[it.sequence_id] = it.severity

    items.sort(
        key=lambda i: (-seq_max[i.sequence_id], i.sequence_id, i.timestamp)
    )
    return items
```

- [ ] **Step 4: Run; confirm pass**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_queue.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/queue.py experiments/data-quality/frame-level/tests/test_review_app_queue.py
git commit -m "feat(data-quality/frame-level): review_app queue with sequence clustering"
```

---

## Task 6: `state.py` — AppState (load + cache)

**Files:**
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/state.py`
- Create: `experiments/data-quality/frame-level/tests/test_review_app_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_review_app_state.py
import json
from pathlib import Path

import pytest

from data_quality_frame_level.review_app.state import AppState, Paths


def _write_predictions(p: Path, frames: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "model_name": "m",
                "split_dir": "data/01_raw/datasets/val",
                "conf_thresh": 0.05,
                "frames": frames,
            }
        )
    )


@pytest.fixture
def fake_tree(tmp_path: Path) -> Paths:
    split = tmp_path / "01_raw" / "datasets" / "val"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir(parents=True)
    stem = "seq_2024-01-01T00-00-00"
    (split / "images" / f"{stem}.jpg").write_bytes(b"jpeg")
    (split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    pred_path = tmp_path / "07_model_output" / "m" / "val" / "predictions.json"
    _write_predictions(
        pred_path,
        {
            stem: {
                "image_path": f"images/{stem}.jpg",
                "predictions": [
                    {
                        "class_id": 0, "cx": 0.5, "cy": 0.5,
                        "w": 0.1, "h": 0.1, "conf": 0.9,
                    }
                ],
            }
        },
    )
    return Paths(
        split_dir=split,
        predictions_path=pred_path,
        review_path=tmp_path / "09_review" / "m" / "val" / "review.json",
    )


def test_load_populates_predictions_and_gt(fake_tree: Paths):
    state = AppState.load(model="m", split="val", paths=fake_tree)
    assert "seq_2024-01-01T00-00-00" in state.predictions
    assert state.predictions["seq_2024-01-01T00-00-00"][0].conf == 0.9
    assert state.gt["seq_2024-01-01T00-00-00"][0].cx == 0.5


def test_save_sample_writes_review_json(fake_tree: Paths):
    from data_quality_frame_level.dataset import BBox
    state = AppState.load(model="m", split="val", paths=fake_tree)
    state.save_sample(
        stem="seq_2024-01-01T00-00-00",
        status="reviewed",
        bboxes=[BBox(class_id=0, cx=0.4, cy=0.4, w=0.2, h=0.2)],
        reviewer="arthur",
        note="moved",
    )
    payload = json.loads(fake_tree.review_path.read_text())
    sample = payload["samples"]["seq_2024-01-01T00-00-00"]
    assert sample["status"] == "reviewed"
    assert sample["bboxes"][0]["cx"] == 0.4
    assert sample["reviewer"] == "arthur"
```

- [ ] **Step 2: Run; confirm fail**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_state.py -v
```

Expected: import error.

- [ ] **Step 3: Implement `state.py`**

```python
# src/data_quality_frame_level/review_app/state.py
"""In-memory state for one ``(model, split)`` context.

Holds the universe of frames (predictions + GT) plus the review state.
Constructed once per context and cached by the FastAPI layer.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from data_quality_frame_level.dataset import BBox, iter_frames
from data_quality_frame_level.inference import PredBBox
from data_quality_frame_level.review_app.persistence import (
    ReviewState,
    SampleReview,
    read_review_state,
    write_review_state,
)


@dataclass(frozen=True)
class Paths:
    split_dir: Path
    predictions_path: Path
    review_path: Path


def _load_predictions(path: Path) -> dict[str, list[PredBBox]]:
    raw = json.loads(path.read_text())
    out: dict[str, list[PredBBox]] = {}
    for stem, frame in raw["frames"].items():
        out[stem] = [
            PredBBox(
                class_id=int(d["class_id"]),
                cx=float(d["cx"]),
                cy=float(d["cy"]),
                w=float(d["w"]),
                h=float(d["h"]),
                conf=float(d["conf"]),
            )
            for d in frame["predictions"]
        ]
    return out


def _load_gt(split_dir: Path) -> dict[str, list[BBox]]:
    return {f.stem: f.gt_bboxes for f in iter_frames(split_dir)}


@dataclass
class AppState:
    model: str
    split: str
    paths: Paths
    predictions: dict[str, list[PredBBox]]
    gt: dict[str, list[BBox]]
    review: ReviewState

    @classmethod
    def load(cls, *, model: str, split: str, paths: Paths) -> "AppState":
        return cls(
            model=model,
            split=split,
            paths=paths,
            predictions=_load_predictions(paths.predictions_path),
            gt=_load_gt(paths.split_dir),
            review=read_review_state(paths.review_path, model=model, split=split),
        )

    def save_sample(
        self,
        *,
        stem: str,
        status: str,
        bboxes: list[BBox],
        reviewer: str | None,
        note: str | None,
    ) -> SampleReview:
        sample = SampleReview(
            status=status,
            bboxes=list(bboxes),
            reviewer=reviewer,
            note=note,
            reviewed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self.review.samples[stem] = sample
        write_review_state(self.paths.review_path, self.review)
        return sample
```

- [ ] **Step 4: Run; confirm pass**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_state.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/state.py experiments/data-quality/frame-level/tests/test_review_app_state.py
git commit -m "feat(data-quality/frame-level): review_app AppState (load + save)"
```

---

## Task 7: `export.py` — diff + manifest

**Files:**
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/export.py`
- Create: `experiments/data-quality/frame-level/tests/test_review_app_export.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_review_app_export.py
import json
from pathlib import Path

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.export import (
    DiffCounts,
    compute_diff,
    export_corrections,
)
from data_quality_frame_level.review_app.persistence import (
    ReviewState,
    SampleReview,
)


def _bb(cx, cy, w=0.1, h=0.1):
    return BBox(class_id=0, cx=cx, cy=cy, w=w, h=h)


def test_compute_diff_added_removed_modified():
    original = [_bb(0.1, 0.1), _bb(0.5, 0.5)]
    corrected = [_bb(0.5, 0.5), _bb(0.9, 0.9)]
    counts = compute_diff(original=original, corrected=corrected)
    assert counts == DiffCounts(added=1, removed=1, modified=0)


def test_compute_diff_modified():
    original = [_bb(0.5, 0.5, w=0.1, h=0.1)]
    corrected = [_bb(0.55, 0.55, w=0.1, h=0.1)]
    counts = compute_diff(original=original, corrected=corrected)
    assert counts.added == 0 and counts.removed == 0 and counts.modified == 1


def test_export_writes_only_changed(tmp_path: Path):
    originals = {
        "stem_a": [_bb(0.5, 0.5)],
        "stem_b": [_bb(0.5, 0.5)],
    }
    review = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(status="reviewed", bboxes=[_bb(0.5, 0.5)]),
            "stem_b": SampleReview(status="reviewed", bboxes=[_bb(0.6, 0.6)]),
            "stem_c": SampleReview(status="unclear", bboxes=[_bb(0.5, 0.5)]),
        },
    )
    out = tmp_path / "10_export" / "m" / "val"
    export_corrections(review=review, originals=originals, out_dir=out)
    assert (out / "labels" / "stem_b.txt").exists()
    assert not (out / "labels" / "stem_a.txt").exists()
    assert not (out / "labels" / "stem_c.txt").exists()
    text = (out / "labels" / "stem_b.txt").read_text()
    assert text.strip().split() == ["0", "0.6", "0.6", "0.1", "0.1"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["totals"]["changed"] == 1
    assert [c["stem"] for c in manifest["changed"]] == ["stem_b"]
```

- [ ] **Step 2: Run; confirm fail**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_export.py -v
```

Expected: import error.

- [ ] **Step 3: Implement `export.py`**

```python
# src/data_quality_frame_level/review_app/export.py
"""Export corrected GT to a YOLO-format patch + manifest.

Only stems whose corrected bboxes differ from the on-disk original are
written. Emits a flat ``labels/<stem>.txt`` tree (no split subdir) and
a ``manifest.json`` summarizing what changed. ``unclear`` samples are
excluded — they are open questions, not decisions.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.matching import iou
from data_quality_frame_level.review_app.persistence import ReviewState

UNCHANGED_IOU = 0.95


@dataclass(frozen=True)
class DiffCounts:
    added: int
    removed: int
    modified: int

    @property
    def is_change(self) -> bool:
        return self.added + self.removed + self.modified > 0


def compute_diff(*, original: list[BBox], corrected: list[BBox]) -> DiffCounts:
    matched_orig: set[int] = set()
    matched_corr: set[int] = set()
    modified = 0
    candidates = sorted(
        (
            (i, j, iou(o, c))
            for i, o in enumerate(original)
            for j, c in enumerate(corrected)
        ),
        key=lambda x: x[2],
        reverse=True,
    )
    for oi, cj, score in candidates:
        if score == 0.0:
            break
        if oi in matched_orig or cj in matched_corr:
            continue
        matched_orig.add(oi)
        matched_corr.add(cj)
        if score < UNCHANGED_IOU:
            modified += 1
    removed = len(original) - len(matched_orig)
    added = len(corrected) - len(matched_corr)
    return DiffCounts(added=added, removed=removed, modified=modified)


def _write_yolo_txt(path: Path, bboxes: list[BBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{b.class_id} {b.cx} {b.cy} {b.w} {b.h}"
        for b in bboxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def export_corrections(
    *,
    review: ReviewState,
    originals: dict[str, list[BBox]],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    changed: list[dict] = []
    totals = {"changed": 0, "added": 0, "removed": 0, "modified": 0}
    for stem in sorted(review.samples):
        sample = review.samples[stem]
        if sample.status != "reviewed":
            continue
        original = originals.get(stem, [])
        diff = compute_diff(original=original, corrected=sample.bboxes)
        if not diff.is_change:
            continue
        _write_yolo_txt(labels_dir / f"{stem}.txt", sample.bboxes)
        changed.append(
            {
                "stem": stem,
                "added": diff.added,
                "removed": diff.removed,
                "modified": diff.modified,
                "reviewer": sample.reviewer,
                "note": sample.note,
            }
        )
        totals["changed"] += 1
        totals["added"] += diff.added
        totals["removed"] += diff.removed
        totals["modified"] += diff.modified
    manifest = {
        "version": 1,
        "model_name": review.model,
        "split": review.split,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "changed": changed,
        "totals": totals,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest
```

- [ ] **Step 4: Run; confirm pass**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_export.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/export.py experiments/data-quality/frame-level/tests/test_review_app_export.py
git commit -m "feat(data-quality/frame-level): review_app export to corrected YOLO + manifest"
```

---

## Task 8: `main.py` — FastAPI app + routes

**Files:**
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/main.py`
- Create: `experiments/data-quality/frame-level/tests/test_review_app_main.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_review_app_main.py
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from data_quality_frame_level.review_app.main import create_app
from data_quality_frame_level.review_app.state import Paths


@pytest.fixture
def app_tree(tmp_path: Path) -> tuple[TestClient, Paths]:
    split = tmp_path / "01_raw" / "datasets" / "val"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir(parents=True)
    stems = ["s_2024-01-01T00-00-00", "s_2024-01-01T00-00-30"]
    for st in stems:
        (split / "images" / f"{st}.jpg").write_bytes(b"jpeg")
        (split / "labels" / f"{st}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    pred_path = tmp_path / "07_model_output" / "m" / "val" / "predictions.json"
    pred_path.parent.mkdir(parents=True)
    pred_path.write_text(
        json.dumps(
            {
                "model_name": "m",
                "split_dir": "data/01_raw/datasets/val",
                "conf_thresh": 0.05,
                "frames": {
                    stems[0]: {
                        "image_path": f"images/{stems[0]}.jpg",
                        "predictions": [
                            {
                                "class_id": 0, "cx": 0.7, "cy": 0.7,
                                "w": 0.1, "h": 0.1, "conf": 0.9,
                            }
                        ],
                    },
                    stems[1]: {
                        "image_path": f"images/{stems[1]}.jpg",
                        "predictions": [],
                    },
                },
            }
        )
    )
    paths = Paths(
        split_dir=split,
        predictions_path=pred_path,
        review_path=tmp_path / "09_review" / "m" / "val" / "review.json",
    )
    app = create_app(
        contexts={("m", "val"): paths},
        models=["m"],
        splits=["val"],
    )
    return TestClient(app), paths


def test_get_contexts(app_tree):
    client, _ = app_tree
    r = client.get("/api/contexts")
    assert r.status_code == 200
    body = r.json()
    assert body["models"] == ["m"]
    assert body["splits"] == ["val"]


def test_get_queue_fp(app_tree):
    client, _ = app_tree
    r = client.get(
        "/api/queue",
        params={
            "model": "m", "split": "val", "view": "fp",
            "conf": 0.05, "iou": 0.05, "review_conf": 0.5,
        },
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert [i["stem"] for i in items] == ["s_2024-01-01T00-00-00"]


def test_get_sample_returns_layers_and_neighbors(app_tree):
    client, _ = app_tree
    r = client.get(
        "/api/sample",
        params={
            "model": "m", "split": "val",
            "stem": "s_2024-01-01T00-00-00",
            "conf": 0.05, "iou": 0.05, "review_conf": 0.5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["original_gt"]) == 1
    assert len(body["predictions"]) == 1
    assert body["sequence_neighbors"][0]["stem"] in {
        "s_2024-01-01T00-00-00", "s_2024-01-01T00-00-30",
    }


def test_post_sample_persists(app_tree):
    client, paths = app_tree
    r = client.post(
        "/api/sample",
        params={"model": "m", "split": "val"},
        json={
            "stem": "s_2024-01-01T00-00-00",
            "status": "reviewed",
            "bboxes": [
                {"class_id": 0, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}
            ],
            "reviewer": "arthur",
            "note": "fixed",
        },
    )
    assert r.status_code == 200
    payload = json.loads(paths.review_path.read_text())
    sample = payload["samples"]["s_2024-01-01T00-00-00"]
    assert sample["status"] == "reviewed"
    assert sample["bboxes"][0]["cx"] == 0.4
```

- [ ] **Step 2: Run; confirm fail**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_main.py -v
```

Expected: import error.

- [ ] **Step 3: Implement `main.py`**

```python
# src/data_quality_frame_level/review_app/main.py
"""FastAPI app for the frame-level review workflow.

Routes:
  GET  /api/contexts                                 — available models + splits
  GET  /api/queue?model&split&view&conf&iou&review_conf  — ordered queue
  GET  /api/sample?model&split&stem&conf&iou&review_conf — layers + neighbors
  POST /api/sample?model&split  (body: SaveBody)     — save corrected GT
  GET  /image?model&split&stem                       — JPEG bytes
  GET  /                                             — static index.html
"""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.matching import evaluate_frame
from data_quality_frame_level.review_app.queue import build_queue
from data_quality_frame_level.review_app.sequence import parse_stem
from data_quality_frame_level.review_app.state import AppState, Paths


class BBoxModel(BaseModel):
    class_id: int = 0
    cx: float
    cy: float
    w: float
    h: float


class PredModel(BBoxModel):
    conf: float
    status: str = "fp"


class SaveBody(BaseModel):
    stem: str
    status: str = Field(..., pattern="^(reviewed|unclear)$")
    bboxes: list[BBoxModel]
    reviewer: str | None = None
    note: str | None = None


def create_app(
    *,
    contexts: dict[tuple[str, str], Paths],
    models: list[str],
    splits: list[str],
) -> FastAPI:
    cache: dict[tuple[str, str], AppState] = {}

    def _state(model: str, split: str) -> AppState:
        key = (model, split)
        if key not in contexts:
            raise HTTPException(404, f"unknown context: {key}")
        if key not in cache:
            cache[key] = AppState.load(model=model, split=split, paths=contexts[key])
        return cache[key]

    app = FastAPI()

    @app.get("/api/contexts")
    def get_contexts() -> dict:
        return {"models": models, "splits": splits}

    @app.get("/api/queue")
    def get_queue(
        model: str, split: str, view: str,
        conf: float, iou: float, review_conf: float,
    ) -> dict:
        s = _state(model, split)
        items = build_queue(
            predictions=s.predictions,
            gt=s.gt,
            review_status={k: v.status for k, v in s.review.samples.items()},
            view=view,
            conf_thresh=conf,
            iou_thresh=iou,
            review_conf_thresh=review_conf,
        )
        return {"items": [asdict(i) for i in items]}

    @app.get("/api/sample")
    def get_sample(
        model: str, split: str, stem: str,
        conf: float, iou: float, review_conf: float,
    ) -> dict:
        s = _state(model, split)
        if stem not in s.gt and stem not in s.predictions:
            raise HTTPException(404, f"unknown stem: {stem}")
        gt = s.gt.get(stem, [])
        preds = [p for p in s.predictions.get(stem, []) if p.conf >= conf]
        ev = evaluate_frame(gt=gt, predictions=preds, iou_thresh=iou)
        sample = s.review.samples.get(stem)
        seq_id, ts = parse_stem(stem)
        neighbors = sorted(
            (
                {"stem": st, "timestamp": parse_stem(st)[1]}
                for st in s.gt.keys() | s.predictions.keys()
                if parse_stem(st)[0] == seq_id
            ),
            key=lambda d: d["timestamp"],
        )
        return {
            "stem": stem,
            "sequence_id": seq_id,
            "timestamp": ts,
            "original_gt": [
                {**asdict(b), "status": st}
                for b, st in zip(gt, ev.gt_status, strict=True)
            ],
            "predictions": [
                {**asdict(p), "status": st}
                for p, st in zip(preds, ev.pred_status, strict=True)
            ],
            "corrected_gt": [asdict(b) for b in (sample.bboxes if sample else [])],
            "status": sample.status if sample else None,
            "reviewer": sample.reviewer if sample else None,
            "note": sample.note if sample else None,
            "reviewed_at": sample.reviewed_at if sample else None,
            "sequence_neighbors": neighbors,
        }

    @app.post("/api/sample")
    def save_sample(model: str, split: str, body: SaveBody) -> dict:
        s = _state(model, split)
        bboxes = [
            BBox(class_id=b.class_id, cx=b.cx, cy=b.cy, w=b.w, h=b.h)
            for b in body.bboxes
        ]
        sample = s.save_sample(
            stem=body.stem,
            status=body.status,
            bboxes=bboxes,
            reviewer=body.reviewer,
            note=body.note,
        )
        return {"saved_at": sample.reviewed_at}

    @app.get("/image")
    def get_image(model: str, split: str, stem: str) -> FileResponse:
        s = _state(model, split)
        path = s.paths.split_dir / "images" / f"{stem}.jpg"
        if not path.is_file():
            raise HTTPException(404, f"missing image: {stem}")
        return FileResponse(path, media_type="image/jpeg")

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app
```

- [ ] **Step 4: Run; confirm pass**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/test_review_app_main.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/main.py experiments/data-quality/frame-level/tests/test_review_app_main.py
git commit -m "feat(data-quality/frame-level): review_app FastAPI routes"
```

---

## Task 9: CLI — `run_review_app.py` + Makefile target

**Files:**
- Create: `experiments/data-quality/frame-level/scripts/run_review_app.py`
- Modify: `experiments/data-quality/frame-level/Makefile`

- [ ] **Step 1: Write the entrypoint**

```python
# scripts/run_review_app.py
"""Launch the frame-level review app via uvicorn.

Discovers contexts from ``params.yaml`` (models) and the
``data/01_raw/datasets/`` tree (splits). Lazy-loads each context on
first request.

Usage::

    uv run --group review-app python scripts/run_review_app.py
"""

import argparse
from pathlib import Path

import uvicorn
import yaml

from data_quality_frame_level.review_app.main import create_app
from data_quality_frame_level.review_app.state import Paths


def _discover_paths(repo_root: Path) -> tuple[
    dict[tuple[str, str], Paths], list[str], list[str]
]:
    params = yaml.safe_load((repo_root / "params.yaml").read_text())
    models = list(params["models"].keys())
    datasets_root = repo_root / "data" / "01_raw" / "datasets"
    splits = sorted(p.name for p in datasets_root.iterdir() if p.is_dir())
    contexts: dict[tuple[str, str], Paths] = {}
    for model in models:
        for split in splits:
            split_dir = datasets_root / split
            pred_path = (
                repo_root / "data" / "07_model_output" / model / split / "predictions.json"
            )
            review_path = (
                repo_root / "data" / "09_review" / model / split / "review.json"
            )
            if pred_path.is_file() and split_dir.is_dir():
                contexts[(model, split)] = Paths(
                    split_dir=split_dir,
                    predictions_path=pred_path,
                    review_path=review_path,
                )
    return contexts, models, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    contexts, models, splits = _discover_paths(args.repo_root)
    app = create_app(contexts=contexts, models=models, splits=splits)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add Makefile target**

In `Makefile`, append:

```make
review-app: ## Launch the bbox-editing review app on http://localhost:8000
	uv run --group review-app python scripts/run_review_app.py
```

Update the `.PHONY` line at the top to include `review-app`.

Also add `pyyaml` to the `review-app` group in `pyproject.toml` (yaml import in run_review_app.py):

```toml
review-app = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "httpx>=0.27",
    "pyyaml>=6.0",
]
```

Then `cd experiments/data-quality/frame-level && uv sync --group review-app`.

- [ ] **Step 3: Smoke-test the entrypoint imports**

```bash
cd experiments/data-quality/frame-level && uv run --group review-app python -c "import scripts.run_review_app"
```

Expected: no output. (If it complains about not finding `scripts`, that's fine — invoke as a file path:
`uv run --group review-app python scripts/run_review_app.py --help` should print help.)

- [ ] **Step 4: Commit**

```bash
git add experiments/data-quality/frame-level/scripts/run_review_app.py experiments/data-quality/frame-level/Makefile experiments/data-quality/frame-level/pyproject.toml experiments/data-quality/frame-level/uv.lock
git commit -m "feat(data-quality/frame-level): review-app make target + uvicorn launcher"
```

---

## Task 10: HTML/CSS scaffold + initial JS (state, API client, dropdowns)

**Files:**
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/index.html`
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.css`
- Create: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js`

- [ ] **Step 1: Write `index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Frame-level review</title>
  <link rel="stylesheet" href="/app.css">
</head>
<body>
  <header id="hdr">
    <strong>label-review</strong>
    <label>model <select id="sel-model"></select></label>
    <label>split <select id="sel-split"></select></label>
    <label>view
      <span class="chips" id="view-chips">
        <button data-view="fp" class="active">FP</button>
        <button data-view="fn">FN</button>
        <button data-view="all">All</button>
      </span>
    </label>
    <span class="grow"></span>
    <label>reviewer <input id="reviewer" placeholder="handle"></label>
    <span id="progress"></span>
  </header>

  <main id="body">
    <aside id="left">
      <section id="filters">
        <h4>FILTERS</h4>
        <div class="slider-row"><span>conf ≥</span><input type="range" id="conf" min="0.05" max="1" step="0.01" value="0.05"><span class="val" id="conf-v">0.05</span></div>
        <div class="slider-row"><span>IoU ≥</span><input type="range" id="iou" min="0" max="1" step="0.01" value="0.05"><span class="val" id="iou-v">0.05</span></div>
        <div class="slider-row"><span>review ≥</span><input type="range" id="review-conf" min="0" max="1" step="0.01" value="0.35"><span class="val" id="review-conf-v">0.35</span></div>
      </section>
      <section id="queue"></section>
    </aside>

    <section id="center">
      <div id="canvas-wrap">
        <canvas id="cnv"></canvas>
        <div id="layer-toggles">
          <label><input type="checkbox" id="show-orig" checked> O original</label>
          <label><input type="checkbox" id="show-pred" checked> P predictions</label>
        </div>
      </div>
      <div id="timeline"></div>
    </section>

    <aside id="right">
      <section id="bbox-list"></section>
      <section id="status-pane">
        <h4>STATUS</h4>
        <div class="opts">
          <button data-status="reviewed" class="active">reviewed</button>
          <button data-status="unclear">unclear</button>
        </div>
      </section>
      <section id="note-pane">
        <h4>NOTE</h4>
        <textarea id="note" rows="3"></textarea>
      </section>
      <div id="save-bar">— no edits —</div>
    </aside>
  </main>

  <script src="/app.js" type="module"></script>
</body>
</html>
```

- [ ] **Step 2: Write `app.css`**

```css
/* static/app.css */
* { box-sizing: border-box; }
body { margin: 0; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; color: #1f2328; }
#hdr { display: flex; gap: 12px; align-items: center; padding: 8px 12px; background: #f6f8fa; border-bottom: 1px solid #d0d7de; }
#hdr select, #hdr input { font: inherit; padding: 3px 6px; border: 1px solid #d0d7de; border-radius: 4px; background: white; }
#hdr .grow { flex: 1; }
#hdr .chips button { font: inherit; padding: 3px 8px; border: 1px solid #d0d7de; border-radius: 12px; background: white; cursor: pointer; margin-right: 2px; }
#hdr .chips button.active { background: #ddf4ff; border-color: #54aeff; color: #0969da; font-weight: 600; }

#body { display: flex; height: calc(100vh - 41px); }

#left { width: 240px; background: #f6f8fa; border-right: 1px solid #d0d7de; display: flex; flex-direction: column; }
#filters { padding: 8px 12px; border-bottom: 1px solid #d0d7de; }
#filters h4 { margin: 0 0 6px; font-size: 10px; color: #57606a; letter-spacing: .05em; }
.slider-row { display: grid; grid-template-columns: 60px 1fr 36px; gap: 6px; align-items: center; margin-bottom: 6px; font-size: 11px; }
.slider-row .val { color: #0969da; font-variant-numeric: tabular-nums; text-align: right; }
#queue { flex: 1; overflow: auto; }
.queue-seq { padding: 5px 12px; font-size: 10px; font-weight: 600; color: #57606a; background: #eaeef2; border-bottom: 1px solid #d0d7de; display: flex; justify-content: space-between; position: sticky; top: 0; }
.queue-item { display: flex; gap: 8px; align-items: center; padding: 6px 12px; border-bottom: 1px solid #eaeef2; cursor: pointer; }
.queue-item:hover { background: #f0f4f8; }
.queue-item.active { background: #ddf4ff; border-left: 3px solid #0969da; padding-left: 9px; }
.queue-item .stem { font-family: ui-monospace, monospace; font-size: 10px; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.queue-item .dot { width: 8px; height: 8px; border-radius: 50%; background: #d0d7de; }
.queue-item .dot.reviewed { background: #1a7f37; }
.queue-item .dot.unclear { background: #d29922; }

#center { flex: 1; display: flex; flex-direction: column; background: #1f2328; min-width: 0; }
#canvas-wrap { flex: 1; position: relative; display: flex; align-items: center; justify-content: center; padding: 16px; min-height: 0; }
#cnv { background: #0d1117; box-shadow: 0 4px 20px rgba(0,0,0,.4); cursor: crosshair; max-width: 100%; max-height: 100%; }
#layer-toggles { position: absolute; top: 8px; right: 8px; color: #c9d1d9; font-size: 11px; display: flex; gap: 12px; background: rgba(0,0,0,.4); padding: 4px 8px; border-radius: 4px; }
#timeline { background: #0d1117; border-top: 1px solid #30363d; padding: 8px 12px; min-height: 70px; display: flex; gap: 4px; overflow-x: auto; color: #8b949e; }
.tl-frame { flex-shrink: 0; cursor: pointer; }
.tl-frame .tl-img { width: 64px; height: 36px; background: #3d444d; border: 1.5px solid transparent; border-radius: 2px; }
.tl-frame.current .tl-img { border-color: #58a6ff; }
.tl-frame .tl-time { font-size: 9px; text-align: center; margin-top: 2px; font-family: ui-monospace, monospace; }

#right { width: 280px; background: #f6f8fa; border-left: 1px solid #d0d7de; display: flex; flex-direction: column; }
#right h4 { margin: 8px 12px 4px; font-size: 10px; color: #57606a; letter-spacing: .05em; }
#bbox-list { padding: 0 12px; flex: 1; overflow: auto; }
.bbox-row { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 4px; margin-bottom: 4px; font-size: 11px; cursor: pointer; }
.bbox-row.orig { background: #ddf4ff; border-left: 3px solid #0969da; }
.bbox-row.corr { background: #dafbe1; border-left: 3px solid #1a7f37; }
.bbox-row.pred { background: #ffebe9; border-left: 3px solid #cf222e; }
.bbox-row .src { font-weight: 600; min-width: 36px; }
.bbox-row .meta-x { color: #57606a; font-family: ui-monospace, monospace; font-size: 10px; flex: 1; }
.bbox-row .actions button { font: inherit; font-size: 10px; padding: 2px 6px; border: 1px solid #d0d7de; border-radius: 3px; background: white; cursor: pointer; }
#status-pane .opts, #right .opts { display: flex; gap: 6px; padding: 0 12px 8px; }
#status-pane .opts button, #right .opts button { flex: 1; font: inherit; padding: 5px; border: 1px solid #d0d7de; border-radius: 4px; background: white; cursor: pointer; }
#status-pane .opts button.active { background: #dafbe1; border-color: #1a7f37; color: #1a7f37; font-weight: 600; }
#note-pane { padding: 0 12px 8px; }
#note { width: 100%; border: 1px solid #d0d7de; border-radius: 4px; padding: 6px; font: inherit; font-size: 11px; resize: vertical; }
#save-bar { padding: 8px 12px; background: #dafbe1; color: #1a7f37; font-weight: 600; text-align: center; border-top: 1px solid #1a7f37; font-size: 11px; }
#save-bar.dirty { background: #fff8c5; color: #9a6700; border-top-color: #9a6700; }
```

- [ ] **Step 3: Write minimal `app.js` (state + API client + dropdowns)**

```javascript
// static/app.js
const state = {
  model: null, split: null,
  view: 'fp',
  conf: 0.05, iou: 0.05, reviewConf: 0.35,
  showOrig: true, showPred: true,
  reviewer: localStorage.getItem('reviewer') || '',
  queue: [], queueIndex: -1,
  sample: null,
  dirty: false,
};

const api = {
  contexts: () => fetch('/api/contexts').then(r => r.json()),
  queue: ({ model, split, view, conf, iou, reviewConf }) =>
    fetch(`/api/queue?model=${model}&split=${split}&view=${view}&conf=${conf}&iou=${iou}&review_conf=${reviewConf}`).then(r => r.json()),
  sample: ({ model, split, stem, conf, iou, reviewConf }) =>
    fetch(`/api/sample?model=${model}&split=${split}&stem=${encodeURIComponent(stem)}&conf=${conf}&iou=${iou}&review_conf=${reviewConf}`).then(r => r.json()),
  save: ({ model, split, body }) =>
    fetch(`/api/sample?model=${model}&split=${split}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json()),
};

async function init() {
  document.getElementById('reviewer').value = state.reviewer;
  document.getElementById('reviewer').addEventListener('input', e => {
    state.reviewer = e.target.value;
    localStorage.setItem('reviewer', state.reviewer);
  });

  const ctx = await api.contexts();
  const selModel = document.getElementById('sel-model');
  const selSplit = document.getElementById('sel-split');
  ctx.models.forEach(m => selModel.add(new Option(m, m)));
  ctx.splits.forEach(s => selSplit.add(new Option(s, s)));
  state.model = ctx.models[0];
  state.split = ctx.splits.includes('val') ? 'val' : ctx.splits[0];
  selModel.value = state.model;
  selSplit.value = state.split;
  selModel.addEventListener('change', () => { state.model = selModel.value; reloadQueue(); });
  selSplit.addEventListener('change', () => { state.split = selSplit.value; reloadQueue(); });

  await reloadQueue();
}

async function reloadQueue() {
  const r = await api.queue({
    model: state.model, split: state.split, view: state.view,
    conf: state.conf, iou: state.iou, reviewConf: state.reviewConf,
  });
  state.queue = r.items;
  state.queueIndex = state.queue.length > 0 ? 0 : -1;
  renderProgress();
  // queue rendering + sample loading wired up in later tasks
}

function renderProgress() {
  const reviewed = state.queue.filter(i => i.status === 'reviewed').length;
  document.getElementById('progress').textContent =
    `${reviewed} / ${state.queue.length} reviewed`;
}

window.addEventListener('DOMContentLoaded', init);
export {};
```

- [ ] **Step 4: Smoke-test in a browser**

```bash
cd experiments/data-quality/frame-level && uv run --group review-app python scripts/run_review_app.py --port 8765 &
sleep 2 && curl -sf http://localhost:8765/api/contexts | head -c 200
kill %1
```

Expected: a JSON body with `models` and `splits`. Browser at `http://localhost:8765` shows the workbench frame with populated dropdowns.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/index.html experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.css experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js
git commit -m "feat(data-quality/frame-level): review-app static scaffold (HTML/CSS/JS)"
```

---

## Task 11: Queue panel rendering + sample loading

**Files:**
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js`

- [ ] **Step 1: Add queue rendering + sample fetch**

Replace the `reloadQueue` function and add `renderQueue` + `loadSample`:

```javascript
async function reloadQueue() {
  await flushPending();
  const r = await api.queue({
    model: state.model, split: state.split, view: state.view,
    conf: state.conf, iou: state.iou, reviewConf: state.reviewConf,
  });
  state.queue = r.items;
  state.queueIndex = state.queue.length > 0 ? 0 : -1;
  renderQueue();
  renderProgress();
  if (state.queueIndex >= 0) await loadSample(state.queue[0].stem);
  else { state.sample = null; renderCanvas(); renderRight(); renderTimeline(); }
}

function renderQueue() {
  const root = document.getElementById('queue');
  root.innerHTML = '';
  let lastSeq = null;
  state.queue.forEach((it, idx) => {
    if (it.sequence_id !== lastSeq) {
      const h = document.createElement('div');
      h.className = 'queue-seq';
      h.innerHTML = `<span>${it.sequence_id}</span><span></span>`;
      root.appendChild(h);
      lastSeq = it.sequence_id;
    }
    const row = document.createElement('div');
    row.className = 'queue-item' + (idx === state.queueIndex ? ' active' : '');
    row.innerHTML = `
      <span class="stem">${it.timestamp}</span>
      <span class="kind">${it.kind}</span>
      <span class="dot ${it.status || ''}"></span>`;
    row.addEventListener('click', () => navigateTo(idx));
    root.appendChild(row);
  });
}

async function navigateTo(idx) {
  if (idx < 0 || idx >= state.queue.length) return;
  await flushPending();
  state.queueIndex = idx;
  renderQueue();
  await loadSample(state.queue[idx].stem);
}

async function loadSample(stem) {
  state.sample = await api.sample({
    model: state.model, split: state.split, stem,
    conf: state.conf, iou: state.iou, reviewConf: state.reviewConf,
  });
  state.dirty = false;
  setSaveBar();
  renderCanvas();
  renderRight();
  renderTimeline();
}

function setSaveBar(text) {
  const b = document.getElementById('save-bar');
  if (text) { b.textContent = text; b.classList.toggle('dirty', state.dirty); return; }
  if (state.dirty) { b.textContent = 'unsaved…'; b.classList.add('dirty'); }
  else if (state.sample?.reviewed_at) { b.textContent = `✓ saved at ${state.sample.reviewed_at}`; b.classList.remove('dirty'); }
  else { b.textContent = '— no edits —'; b.classList.remove('dirty'); }
}

// Stubs filled in by later tasks:
function renderCanvas() {}
function renderRight() {}
function renderTimeline() {}
async function flushPending() {}
```

- [ ] **Step 2: Smoke-test**

Reload the browser. Expected: queue panel populates with sequence headers and items. Clicking an item changes the active highlight and the URL stays the same. The save bar reads `— no edits —`.

- [ ] **Step 3: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js
git commit -m "feat(data-quality/frame-level): review-app queue panel + sample loader"
```

---

## Task 12: Canvas — three-layer renderer + bbox interactions

**Files:**
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js`

- [ ] **Step 1: Replace `renderCanvas` and add interactions**

```javascript
const cnv = document.getElementById('cnv');
const ctx = cnv.getContext('2d');
let img = new Image();
let imgLoaded = false;
let selected = null;  // {layer:'corr'|'orig', idx:number} or null
let drag = null;      // {kind:'move'|'resize-tl'|...|'draw', start:{x,y}, ref:bbox|null}

function bboxArea(b) { return b.w * b.h; }
function bboxToRect(b, W, H) {
  return { x: (b.cx - b.w / 2) * W, y: (b.cy - b.h / 2) * H, w: b.w * W, h: b.h * H };
}
function rectToBbox(r, W, H) {
  return { class_id: 0, cx: (r.x + r.w / 2) / W, cy: (r.y + r.h / 2) / H, w: r.w / W, h: r.h / H };
}
function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function clampBbox(b) {
  const w = clamp01(b.w), h = clamp01(b.h);
  const cx = clamp01(b.cx), cy = clamp01(b.cy);
  return { class_id: 0, cx, cy, w: Math.min(w, 2*cx, 2*(1-cx)), h: Math.min(h, 2*cy, 2*(1-cy)) };
}

function renderCanvas() {
  if (!state.sample) { ctx.clearRect(0, 0, cnv.width, cnv.height); return; }
  const url = `/image?model=${state.model}&split=${state.split}&stem=${encodeURIComponent(state.sample.stem)}`;
  if (img.src !== location.origin + url) {
    imgLoaded = false;
    img.onload = () => { imgLoaded = true; sizeCanvas(); paint(); };
    img.src = url;
  } else if (imgLoaded) { sizeCanvas(); paint(); }
}

function sizeCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  const maxW = wrap.clientWidth - 32, maxH = wrap.clientHeight - 32;
  const ar = img.naturalWidth / img.naturalHeight;
  let w = maxW, h = maxW / ar;
  if (h > maxH) { h = maxH; w = maxH * ar; }
  cnv.width = w; cnv.height = h;
}

function paint() {
  ctx.clearRect(0, 0, cnv.width, cnv.height);
  ctx.drawImage(img, 0, 0, cnv.width, cnv.height);
  const W = cnv.width, H = cnv.height;
  const corrected = state.sample.corrected_gt;
  const matchesOrig = (orig) => corrected.some(c => Math.abs(c.cx-orig.cx)<1e-6 && Math.abs(c.cy-orig.cy)<1e-6 && Math.abs(c.w-orig.w)<1e-6 && Math.abs(c.h-orig.h)<1e-6);
  if (state.showOrig) {
    state.sample.original_gt.forEach((b, i) => {
      const removed = state.sample.status === 'reviewed' && !matchesOrig(b) && !corrected.find(c => bboxOverlaps(c, b));
      drawBox(b, { stroke: '#58a6ff', fill: 'rgba(88,166,255,.12)', dashed: false, label: `GT (orig)${removed ? ' · removed' : ''}`, struck: removed, selected: selected?.layer==='orig' && selected.idx===i });
    });
  }
  if (state.showPred) {
    state.sample.predictions.forEach(p => {
      drawBox(p, { stroke: '#f85149', fill: 'transparent', dashed: true, label: `pred · ${p.status} · ${p.conf.toFixed(2)}` });
    });
  }
  corrected.forEach((b, i) => {
    drawBox(b, { stroke: '#3fb950', fill: 'rgba(63,185,80,.10)', dashed: false, label: 'GT (corr)', selected: selected?.layer==='corr' && selected.idx===i, handles: true });
  });
}

function bboxOverlaps(a, b) {
  return Math.abs(a.cx-b.cx) < 0.5*(a.w+b.w) && Math.abs(a.cy-b.cy) < 0.5*(a.h+b.h);
}

function drawBox(b, { stroke, fill, dashed, label, struck=false, selected=false, handles=false }) {
  const r = bboxToRect(b, cnv.width, cnv.height);
  ctx.lineWidth = selected ? 3 : 2;
  ctx.strokeStyle = stroke;
  ctx.fillStyle = fill;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.fillRect(r.x, r.y, r.w, r.h);
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  if (struck) {
    ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(r.x+r.w, r.y+r.h);
    ctx.moveTo(r.x+r.w, r.y); ctx.lineTo(r.x, r.y+r.h); ctx.stroke();
  }
  ctx.setLineDash([]);
  if (label) {
    ctx.font = '11px ui-sans-serif, system-ui';
    const tw = ctx.measureText(label).width + 6;
    ctx.fillStyle = stroke;
    ctx.fillRect(r.x, r.y - 14, tw, 14);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, r.x + 3, r.y - 3);
  }
  if (handles) {
    const corners = [[r.x, r.y], [r.x+r.w, r.y], [r.x, r.y+r.h], [r.x+r.w, r.y+r.h]];
    ctx.fillStyle = stroke;
    corners.forEach(([x, y]) => { ctx.fillRect(x-4, y-4, 8, 8); });
  }
}

// hit-testing + drag
function hit(x, y) {
  for (let i = state.sample.corrected_gt.length - 1; i >= 0; i--) {
    const r = bboxToRect(state.sample.corrected_gt[i], cnv.width, cnv.height);
    const handle = handleAt(r, x, y);
    if (handle) return { layer: 'corr', idx: i, handle };
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
      return { layer: 'corr', idx: i, handle: 'move' };
  }
  if (state.showOrig) {
    for (let i = state.sample.original_gt.length - 1; i >= 0; i--) {
      const r = bboxToRect(state.sample.original_gt[i], cnv.width, cnv.height);
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
        return { layer: 'orig', idx: i, handle: 'click' };
    }
  }
  return null;
}

function handleAt(r, x, y) {
  const tol = 6;
  if (Math.abs(x - r.x) < tol && Math.abs(y - r.y) < tol) return 'tl';
  if (Math.abs(x - (r.x + r.w)) < tol && Math.abs(y - r.y) < tol) return 'tr';
  if (Math.abs(x - r.x) < tol && Math.abs(y - (r.y + r.h)) < tol) return 'bl';
  if (Math.abs(x - (r.x + r.w)) < tol && Math.abs(y - (r.y + r.h)) < tol) return 'br';
  return null;
}

cnv.addEventListener('mousedown', e => {
  if (!state.sample || !imgLoaded) return;
  const rect = cnv.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const h = hit(x, y);
  if (h?.layer === 'orig') {
    // Promote original → corrected and select the new corrected.
    const original = state.sample.original_gt[h.idx];
    state.sample.corrected_gt.push({ class_id: 0, cx: original.cx, cy: original.cy, w: original.w, h: original.h });
    selected = { layer: 'corr', idx: state.sample.corrected_gt.length - 1 };
    markDirty(); paint(); renderRight(); return;
  }
  if (h?.layer === 'corr') {
    selected = { layer: 'corr', idx: h.idx };
    drag = { kind: h.handle, start: { x, y }, ref: { ...state.sample.corrected_gt[h.idx] } };
    paint(); return;
  }
  // Empty space: start drawing a new bbox.
  selected = null;
  drag = { kind: 'draw', start: { x, y }, anchor: { x, y } };
  paint();
});

cnv.addEventListener('mousemove', e => {
  if (!drag || !imgLoaded) return;
  const rect = cnv.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const W = cnv.width, H = cnv.height;
  if (drag.kind === 'draw') {
    paint();
    ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
    ctx.strokeRect(Math.min(drag.start.x, x), Math.min(drag.start.y, y), Math.abs(x-drag.start.x), Math.abs(y-drag.start.y));
    ctx.setLineDash([]);
    return;
  }
  if (drag.kind === 'move') {
    const dx = (x - drag.start.x) / W, dy = (y - drag.start.y) / H;
    state.sample.corrected_gt[selected.idx] = clampBbox({ ...drag.ref, cx: drag.ref.cx + dx, cy: drag.ref.cy + dy });
    paint(); return;
  }
  // resize: tl/tr/bl/br
  const r0 = bboxToRect(drag.ref, W, H);
  let nx = r0.x, ny = r0.y, nw = r0.w, nh = r0.h;
  if (drag.kind === 'tl') { nw = r0.x + r0.w - x; nh = r0.y + r0.h - y; nx = x; ny = y; }
  if (drag.kind === 'tr') { nw = x - r0.x; nh = r0.y + r0.h - y; ny = y; }
  if (drag.kind === 'bl') { nw = r0.x + r0.w - x; nh = y - r0.y; nx = x; }
  if (drag.kind === 'br') { nw = x - r0.x; nh = y - r0.y; }
  if (nw < 4 || nh < 4) return;
  state.sample.corrected_gt[selected.idx] = clampBbox(rectToBbox({ x: nx, y: ny, w: nw, h: nh }, W, H));
  paint();
});

cnv.addEventListener('mouseup', e => {
  if (!drag) return;
  if (drag.kind === 'draw') {
    const rect = cnv.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const W = cnv.width, H = cnv.height;
    const r = { x: Math.min(drag.start.x, x), y: Math.min(drag.start.y, y), w: Math.abs(x-drag.start.x), h: Math.abs(y-drag.start.y) };
    if (r.w >= 4 && r.h >= 4) {
      state.sample.corrected_gt.push(clampBbox(rectToBbox(r, W, H)));
      selected = { layer: 'corr', idx: state.sample.corrected_gt.length - 1 };
      markDirty();
    }
  } else if (drag.kind === 'move' || ['tl','tr','bl','br'].includes(drag.kind)) {
    markDirty();
  }
  drag = null;
  paint();
  renderRight();
});

function markDirty() { state.dirty = true; setSaveBar(); scheduleSave(); }

let saveTimer = null;
function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(persistSample, 1000);
}

async function persistSample() {
  if (!state.dirty || !state.sample) return;
  await api.save({
    model: state.model, split: state.split,
    body: {
      stem: state.sample.stem,
      status: state.sample.status || 'reviewed',
      bboxes: state.sample.corrected_gt,
      reviewer: state.reviewer || null,
      note: state.sample.note || null,
    },
  });
  state.dirty = false;
  state.sample.status = state.sample.status || 'reviewed';
  state.sample.reviewed_at = new Date().toISOString();
  setSaveBar();
  // refresh queue status badges without reloading
  const qi = state.queue.find(q => q.stem === state.sample.stem);
  if (qi) qi.status = state.sample.status;
  renderQueue();
  renderProgress();
}

async function flushPending() {
  if (state.dirty) await persistSample();
  clearTimeout(saveTimer);
}

window.addEventListener('resize', () => imgLoaded && (sizeCanvas(), paint()));
```

- [ ] **Step 2: Smoke-test in a browser**

Reload `http://localhost:8000`. Expected:
- Sample image loads with three layers visible (blue original, red dashed predictions, no green corrected initially).
- Click empty space and drag → new green corrected box appears, save bar turns yellow `unsaved…`, then green `✓ saved at …` after ~1s.
- Click an original blue box → it copies to a green corrected box and becomes selected.
- Drag green box body or corner → moves/resizes, auto-saves.
- Reload page → state persists.

- [ ] **Step 3: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js
git commit -m "feat(data-quality/frame-level): review-app three-layer canvas + bbox edit"
```

---

## Task 13: Right panel — bbox list, status, note

**Files:**
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js`

- [ ] **Step 1: Implement `renderRight` and wire status/note**

```javascript
function renderRight() {
  const root = document.getElementById('bbox-list');
  root.innerHTML = '';
  if (!state.sample) return;
  const make = (cls, src, meta, actions = '') => {
    const row = document.createElement('div');
    row.className = `bbox-row ${cls}`;
    row.innerHTML = `<span class="src">${src}</span><span class="meta-x">${meta}</span><span class="actions">${actions}</span>`;
    return row;
  };
  state.sample.original_gt.forEach((b, i) => {
    const row = make('orig', `GT #${i}`, `${b.cx.toFixed(2)} ${b.cy.toFixed(2)} · ${b.status}`, `<button data-act="promote-orig" data-i="${i}">→ correct</button>`);
    root.appendChild(row);
  });
  state.sample.corrected_gt.forEach((b, i) => {
    const row = make('corr', `corr #${i}`, `${b.cx.toFixed(2)} ${b.cy.toFixed(2)}`, `<button data-act="del-corr" data-i="${i}">✕</button>`);
    root.appendChild(row);
  });
  state.sample.predictions.forEach((p, i) => {
    const row = make('pred', 'pred', `${p.cx.toFixed(2)} ${p.cy.toFixed(2)} · ${p.status} · ${p.conf.toFixed(2)}`, `<button data-act="promote-pred" data-i="${i}">→ correct</button>`);
    root.appendChild(row);
  });
  // status buttons
  document.querySelectorAll('#status-pane button[data-status]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.status === (state.sample.status || 'reviewed'));
    btn.onclick = () => {
      state.sample.status = btn.dataset.status;
      markDirty();
      document.querySelectorAll('#status-pane button[data-status]').forEach(b => b.classList.toggle('active', b === btn));
    };
  });
  // note
  const note = document.getElementById('note');
  note.value = state.sample.note || '';
  note.oninput = () => { state.sample.note = note.value || null; markDirty(); };
}

document.getElementById('bbox-list').addEventListener('click', e => {
  const btn = e.target.closest('button[data-act]'); if (!btn) return;
  const i = +btn.dataset.i;
  if (btn.dataset.act === 'promote-orig') {
    const o = state.sample.original_gt[i];
    state.sample.corrected_gt.push({ class_id: 0, cx: o.cx, cy: o.cy, w: o.w, h: o.h });
  } else if (btn.dataset.act === 'promote-pred') {
    const p = state.sample.predictions[i];
    state.sample.corrected_gt.push({ class_id: 0, cx: p.cx, cy: p.cy, w: p.w, h: p.h });
  } else if (btn.dataset.act === 'del-corr') {
    state.sample.corrected_gt.splice(i, 1);
    if (selected?.layer === 'corr' && selected.idx === i) selected = null;
  }
  markDirty(); paint(); renderRight();
});
```

- [ ] **Step 2: Smoke-test**

Reload. Expected:
- Right panel lists original GT + corrected GT + predictions with the correct color band.
- `→ correct` on a prediction adds a green box on canvas at the prediction's geometry.
- `✕` on a corrected row removes it from the canvas.
- Status buttons toggle active styling; note textarea typing triggers auto-save.

- [ ] **Step 3: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js
git commit -m "feat(data-quality/frame-level): review-app right panel (bboxes/status/note)"
```

---

## Task 14: Timeline strip + filter sliders + view chips + keyboard

**Files:**
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js`

- [ ] **Step 1: Implement `renderTimeline`**

```javascript
function renderTimeline() {
  const root = document.getElementById('timeline');
  root.innerHTML = '';
  if (!state.sample) return;
  const neighbors = state.sample.sequence_neighbors;
  const currentIdx = neighbors.findIndex(n => n.stem === state.sample.stem);
  const start = Math.max(0, currentIdx - 5);
  const end = Math.min(neighbors.length, currentIdx + 6);
  for (let i = start; i < end; i++) {
    const n = neighbors[i];
    const f = document.createElement('div');
    f.className = 'tl-frame' + (n.stem === state.sample.stem ? ' current' : '');
    f.innerHTML = `
      <img class="tl-img" src="/image?model=${state.model}&split=${state.split}&stem=${encodeURIComponent(n.stem)}" alt="">
      <div class="tl-time">${n.timestamp}</div>`;
    f.addEventListener('click', () => loadSample(n.stem));
    root.appendChild(f);
  }
}
```

(In CSS, `.tl-img` is a `<div>`; switch the markup to `<img class="tl-img">` and update CSS to size `img.tl-img` like the placeholder div: `width:64px;height:36px;object-fit:cover;`. Adjust the existing `app.css` rule.)

- [ ] **Step 2: Wire filter sliders + view chips + layer toggles + reviewer field**

Append to `init()`:

```javascript
  ['conf', 'iou', 'review-conf'].forEach(id => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(`${id}-v`);
    el.addEventListener('input', () => {
      const v = parseFloat(el.value);
      valEl.textContent = v.toFixed(2);
      if (id === 'conf') state.conf = v;
      else if (id === 'iou') state.iou = v;
      else state.reviewConf = v;
      debounceReload();
    });
  });
  document.querySelectorAll('#view-chips button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#view-chips button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.view = btn.dataset.view;
      reloadQueue();
    });
  });
  document.getElementById('show-orig').addEventListener('change', e => {
    state.showOrig = e.target.checked; paint();
  });
  document.getElementById('show-pred').addEventListener('change', e => {
    state.showPred = e.target.checked; paint();
  });

let reloadTimer = null;
function debounceReload() {
  clearTimeout(reloadTimer);
  reloadTimer = setTimeout(reloadQueue, 200);
}
```

- [ ] **Step 3: Keyboard shortcuts**

Add at the bottom of `app.js`:

```javascript
window.addEventListener('keydown', async e => {
  if (e.target.matches('input, textarea')) return;
  if (e.key === 'ArrowLeft' && e.ctrlKey) return seqStep(-1);
  if (e.key === 'ArrowRight' && e.ctrlKey) return seqStep(+1);
  if (e.key === 'ArrowLeft') return navigateTo(state.queueIndex - 1);
  if (e.key === 'ArrowRight') return navigateTo(state.queueIndex + 1);
  if (e.key === 'Delete' || e.key === 'Backspace') return deleteSelected();
  if (e.key === 'Escape') { selected = null; paint(); }
  if (e.key === 'r') return setStatus('reviewed');
  if (e.key === 'u') return setStatus('unclear');
  if (e.key === 'o') {
    state.showOrig = !state.showOrig;
    document.getElementById('show-orig').checked = state.showOrig; paint();
  }
  if (e.key === 'p') {
    state.showPred = !state.showPred;
    document.getElementById('show-pred').checked = state.showPred; paint();
  }
});

async function seqStep(d) {
  if (!state.sample) return;
  await flushPending();
  const ns = state.sample.sequence_neighbors;
  const i = ns.findIndex(n => n.stem === state.sample.stem);
  const target = ns[i + d];
  if (target) await loadSample(target.stem);
}

function deleteSelected() {
  if (!state.sample || !selected) return;
  if (selected.layer === 'corr') {
    state.sample.corrected_gt.splice(selected.idx, 1);
    selected = null;
    markDirty(); paint(); renderRight();
  }
}

function setStatus(s) {
  if (!state.sample) return;
  state.sample.status = s;
  markDirty();
  renderRight();
}
```

- [ ] **Step 4: Smoke-test**

Reload. Expected:
- Timeline strip shows ±5 sibling frames with the current one bordered blue.
- Sliders: dragging `conf`/`IoU`/`review` updates the queue length live (debounced 200ms).
- View chips switch FP/FN/All and reload queue.
- ←/→ steps queue, Ctrl ←/→ steps sequence (even into unflagged frames).
- Delete removes the selected corrected box.
- `o`/`p` toggle layers.

- [ ] **Step 5: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.css
git commit -m "feat(data-quality/frame-level): review-app timeline, filters, keyboard"
```

---

## Task 15: Export CLI — `export_review_app.py` + Makefile target

**Files:**
- Create: `experiments/data-quality/frame-level/scripts/export_review_app.py`
- Modify: `experiments/data-quality/frame-level/Makefile`

- [ ] **Step 1: Write the CLI**

```python
# scripts/export_review_app.py
"""Build YOLO-format patches from review.json under data/10_export/.

Iterates every (model, split) for which a review.json exists; emits
labels/<stem>.txt + manifest.json under
``data/10_export/<model>/<split>/``. Existing exports are overwritten.
"""

import argparse
import logging
from pathlib import Path

import yaml

from data_quality_frame_level.dataset import iter_frames
from data_quality_frame_level.review_app.export import export_corrections
from data_quality_frame_level.review_app.persistence import read_review_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root
    params = yaml.safe_load((repo / "params.yaml").read_text())
    models = list(params["models"].keys())
    datasets_root = repo / "data" / "01_raw" / "datasets"
    splits = sorted(p.name for p in datasets_root.iterdir() if p.is_dir())
    for model in models:
        for split in splits:
            review_path = repo / "data" / "09_review" / model / split / "review.json"
            if not review_path.is_file():
                log.info("skip: no review.json at %s", review_path)
                continue
            state = read_review_state(review_path, model=model, split=split)
            originals = {f.stem: f.gt_bboxes for f in iter_frames(datasets_root / split)}
            out_dir = repo / "data" / "10_export" / model / split
            manifest = export_corrections(
                review=state, originals=originals, out_dir=out_dir
            )
            log.info(
                "%s/%s: %d changed, %d added, %d removed, %d modified",
                model, split,
                manifest["totals"]["changed"],
                manifest["totals"]["added"],
                manifest["totals"]["removed"],
                manifest["totals"]["modified"],
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add Makefile target**

In `Makefile`, append:

```make
review-export: ## Build YOLO-format patches under data/10_export/ from review.json
	uv run --group review-app python scripts/export_review_app.py
	@echo ""
	@echo "Next steps to share the export:"
	@echo "  uv run dvc add data/10_export && uv run dvc push"
	@echo "  git add data/10_export.dvc data/.gitignore && git commit -m 'export: refresh corrections'"
```

Update `.PHONY` to include `review-export`.

- [ ] **Step 3: Smoke-test (local data permitting)**

```bash
cd experiments/data-quality/frame-level && make review-export
ls data/10_export/yolo11s-nimble-narwhal/val/labels/ | head
cat data/10_export/yolo11s-nimble-narwhal/val/manifest.json | head -40
```

Expected: a (possibly empty if no corrections yet) `labels/` dir and a manifest with `totals`. If `review.json` is empty, `manifest.totals.changed == 0`.

- [ ] **Step 4: Commit**

```bash
git add experiments/data-quality/frame-level/scripts/export_review_app.py experiments/data-quality/frame-level/Makefile
git commit -m "feat(data-quality/frame-level): review-export make target + CLI"
```

---

## Task 16: End-to-end smoke test + lint/format pass

**Files:** none new

- [ ] **Step 1: Run the full pytest suite**

```bash
cd experiments/data-quality/frame-level && uv run pytest tests/ -v
```

Expected: every existing test passes plus the 6 new test files (sequence, matching, persistence, queue, state, export, main).

- [ ] **Step 2: Lint + format**

```bash
cd experiments/data-quality/frame-level && uv run ruff check . && uv run ruff format --check .
```

Fix any issues (`ruff format .` then `ruff check . --fix` for auto-fixable).

- [ ] **Step 3: End-to-end smoke**

```bash
cd experiments/data-quality/frame-level && uv run --group review-app python scripts/run_review_app.py --port 8765
```

In a browser at `http://localhost:8765`:

1. Header dropdowns populate with `yolo11s-nimble-narwhal` and `train`/`val`/`test`.
2. Selecting `val` + view `FP` shows a queue with sequence headers and items in the order matching `params.yaml` defaults (conf 0.05, IoU 0.05, review 0.35).
3. The first sample loads — image renders with blue original GT (if any), red dashed predictions, no green initially. Timeline strip shows ±5 siblings.
4. Drag confidence slider down → queue grows; drag back up → queue shrinks.
5. Click a prediction's `→ correct` → a green box appears at that geometry; save bar transitions yellow → green.
6. Press `r` → status set to `reviewed`; queue dot turns green.
7. Press `→` → next queue item loads.
8. Press `Ctrl →` → next *sequence* sibling loads (which may not be flagged).
9. Press `o` then `p` → original and prediction layers toggle off.
10. Stop the server, re-launch, reload page — corrected GT and status persist (loaded from `review.json`).

- [ ] **Step 4: Build the export**

```bash
cd experiments/data-quality/frame-level && make review-export
```

Verify `data/10_export/yolo11s-nimble-narwhal/val/manifest.json` lists the stem you edited above with `added: 1` (or appropriate counts).

- [ ] **Step 5: Final commit**

If anything was tweaked during smoke testing:

```bash
git add -p experiments/data-quality/frame-level/
git commit -m "fix(data-quality/frame-level): smoke-test polish for review app"
```

Otherwise the work is done.

---

## Self-review notes

- Spec §5.1 (review.json shape) → Tasks 4, 6.
- Spec §5.2 (export shape) → Tasks 7, 15.
- Spec §6.1 (header) → Task 10.
- Spec §6.2 (filters, live recompute) → Tasks 8, 14.
- Spec §6.3 (queue with sequence clustering) → Tasks 5, 11.
- Spec §6.4 (three-layer canvas) → Task 12.
- Spec §6.5 (timeline strip) → Task 14.
- Spec §6.6 (right panel) → Task 13.
- Spec §6.7 (keyboard shortcuts) → Task 14.
- Spec §7 (atomic auto-save) → Tasks 4, 12.
- Spec §8 (live IoU recompute) → Tasks 3, 8, 14.
- Spec §9 (tech stack + file layout) → Task 1, layout up top.
- Spec §10 (FiftyOne coexistence) → unchanged code paths; verified by Task 16.

No spec section is unaddressed. Type names referenced consistently across tasks (`BBox`, `PredBBox`, `SampleReview`, `ReviewState`, `Paths`, `AppState`, `QueueItem`, `EvaluatedFrame`, `DiffCounts`).

# Smoke Tube Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-sequence smoke-tube dataset (one tube per sequence, binary smoke/fp label, metadata-only JSON) by reading existing label `.txt` files directly — no YOLO inference pass.

**Architecture:** Read 5-col GT (WF) and 6-col YOLO predictions (FP) from `data/01_raw/datasets/{train,val}/{wildfire,fp}/<seq>/labels/*.txt`. Build candidate tubes via existing `build_tubes` IoU matching, select the longest, geometrically interpolate gap bboxes, filter, write one JSON per surviving sequence to `data/03_primary/tubes/{train,val}/`. Wire as a DVC stage.

**Tech Stack:** Python 3.11+, uv, pytest, ruff, DVC. All commands run with `uv run`. All work happens in `experiments/temporal-models/smokeynet-adapted/`. Spec: `docs/specs/2026-04-14-smoke-tube-dataset-design.md`.

**User commit preference:** Never include `Co-Authored-By: Claude` or any mention of Claude/Anthropic in commit messages.

**Working directory for ALL commands below:** `experiments/temporal-models/smokeynet-adapted/`

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/smokeynet_adapted/types.py` | Modify | Add `is_gap: bool = False` to `TubeEntry` |
| `src/smokeynet_adapted/data.py` | Modify | Add `load_detections`, `load_frame_detections` |
| `src/smokeynet_adapted/tubes.py` | Modify | Add `select_longest_tube`, `interpolate_gaps` |
| `tests/test_data.py` (new) | Create | Tests for `load_detections` (3 cases) |
| `tests/test_tubes.py` | Modify | Tests for `select_longest_tube`, `interpolate_gaps` |
| `scripts/build_tubes.py` (new) | Create | CLI: walk sequences → tube JSONs + summary |
| `params.yaml` | Modify | Add `build_tubes` section |
| `dvc.yaml` | Modify | Add `build_tubes` foreach stage |
| `notebooks/02-visualize-built-tubes.ipynb` (new) | Create | Inspect persisted tubes |

---

## Task 1: Extend `TubeEntry` with `is_gap`

**Files:**
- Modify: `src/smokeynet_adapted/types.py`

- [ ] **Step 1: Add the field**

Edit `src/smokeynet_adapted/types.py`. Find the `TubeEntry` dataclass:

```python
@dataclass
class TubeEntry:
    """A single entry in a smoke tube.

    When ``detection`` is ``None``, this entry represents a gap frame where
    YOLO did not detect the tracked region.  Gap features are filled via
    linear interpolation at the feature level.
    """

    frame_idx: int
    detection: Detection | None = None
```

Replace with:

```python
@dataclass
class TubeEntry:
    """A single entry in a smoke tube.

    ``is_gap`` flags entries whose ``detection`` was not observed by the
    detector and was instead filled in by gap interpolation. After
    interpolation, gap entries always have a ``Detection`` (lerped bbox,
    confidence=0.0); pre-interpolation gaps have ``detection=None``.
    """

    frame_idx: int
    detection: Detection | None = None
    is_gap: bool = False
```

- [ ] **Step 2: Run the existing test suite to make sure nothing broke**

Run: `uv run pytest tests/test_types.py tests/test_tubes.py -v`
Expected: all existing tests PASS (the new field has a default, so existing constructor calls still work).

- [ ] **Step 3: Commit**

```bash
git add src/smokeynet_adapted/types.py
git commit -m "feat(types): add is_gap flag to TubeEntry"
```

---

## Task 2: `load_detections` — unified 5-col / 6-col label reader

**Files:**
- Modify: `src/smokeynet_adapted/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data.py`:

```python
"""Tests for label-file parsing."""

from pathlib import Path

import pytest

from smokeynet_adapted.data import load_detections


def _write_label(tmp_path: Path, frame_id: str, text: str) -> Path:
    seq = tmp_path / "seq_a"
    (seq / "labels").mkdir(parents=True, exist_ok=True)
    (seq / "labels" / f"{frame_id}.txt").write_text(text)
    return seq


def test_load_detections_5col_sets_confidence_to_one(tmp_path):
    seq = _write_label(tmp_path, "f1", "0 0.5 0.4 0.1 0.2\n")
    dets = load_detections(seq, "f1")
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 0
    assert d.cx == pytest.approx(0.5)
    assert d.cy == pytest.approx(0.4)
    assert d.w == pytest.approx(0.1)
    assert d.h == pytest.approx(0.2)
    assert d.confidence == pytest.approx(1.0)


def test_load_detections_6col_reads_confidence_from_last_column(tmp_path):
    seq = _write_label(tmp_path, "f1", "0 0.25 0.30 0.05 0.07 0.42\n")
    dets = load_detections(seq, "f1")
    assert len(dets) == 1
    assert dets[0].confidence == pytest.approx(0.42)


def test_load_detections_empty_file_returns_empty_list(tmp_path):
    seq = _write_label(tmp_path, "f1", "")
    assert load_detections(seq, "f1") == []


def test_load_detections_missing_file_returns_empty_list(tmp_path):
    seq = _write_label(tmp_path, "f1", "")
    assert load_detections(seq, "nonexistent") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: ImportError (`load_detections` not defined).

- [ ] **Step 3: Implement `load_detections`**

Edit `src/smokeynet_adapted/data.py`. Add this import at the top (alongside existing imports):

```python
from .types import Detection
```

Then append the function to the file:

```python
def load_detections(sequence_dir: Path, frame_id: str) -> list[Detection]:
    """Read a YOLO-format label file as :class:`Detection` objects.

    Supports both formats found in the Pyronear dataset:

    * **5-col** ``class cx cy w h`` — wildfire ground-truth annotations.
      ``confidence`` is set to ``1.0``.
    * **6-col** ``class cx cy w h conf`` — false-positive YOLO predictions.
      ``confidence`` is read from the last column.

    Malformed lines (wrong column count, non-numeric values) are silently
    skipped — they should be rare and we don't want one bad line to drop
    a whole sequence.

    Args:
        sequence_dir: Path to the sequence directory (contains ``labels/``).
        frame_id: Frame filename stem.

    Returns:
        List of detections in file order. Empty list if the file is missing
        or empty.
    """
    label_path = sequence_dir / "labels" / f"{frame_id}.txt"
    if not label_path.is_file():
        return []
    content = label_path.read_text().strip()
    if not content:
        return []
    dets: list[Detection] = []
    for line in content.split("\n"):
        parts = line.strip().split()
        try:
            if len(parts) == 5:
                class_id = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:5])
                confidence = 1.0
            elif len(parts) == 6:
                class_id = int(parts[0])
                cx, cy, w, h, confidence = (float(p) for p in parts[1:6])
            else:
                continue
        except ValueError:
            continue
        dets.append(
            Detection(
                class_id=class_id,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                confidence=confidence,
            )
        )
    return dets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Lint check**

Run: `uv run ruff check src/smokeynet_adapted/data.py tests/test_data.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/smokeynet_adapted/data.py tests/test_data.py
git commit -m "feat(data): add load_detections for 5-col GT and 6-col YOLO labels"
```

---

## Task 3: `load_frame_detections` — sequence-level loader

**Files:**
- Modify: `src/smokeynet_adapted/data.py`

No tests for this one — it's a thin loop over `load_detections` and `get_sorted_frames`, both of which are exercised elsewhere. The build_tubes script will be the integration check.

- [ ] **Step 1: Implement**

Edit `src/smokeynet_adapted/data.py`. Append:

```python
def load_frame_detections(sequence_dir: Path) -> list[FrameDetections]:
    """Load all per-frame detections for a sequence in temporal order.

    Iterates frames returned by :func:`get_sorted_frames` (which sorts
    by image filename / timestamp) and reads the corresponding label
    file via :func:`load_detections`.

    Args:
        sequence_dir: Path to the sequence directory.

    Returns:
        Ordered list of :class:`FrameDetections`, one per image. Frames
        with no labels yield an entry with an empty ``detections`` list.
    """
    frame_paths = get_sorted_frames(sequence_dir)
    return [
        FrameDetections(
            frame_idx=idx,
            frame_id=fpath.stem,
            timestamp=parse_timestamp(fpath.stem),
            detections=load_detections(sequence_dir, fpath.stem),
        )
        for idx, fpath in enumerate(frame_paths)
    ]
```

Add `FrameDetections` to the import at the top of the file:

```python
from .types import Detection, FrameDetections
```

- [ ] **Step 2: Lint check**

Run: `uv run ruff check src/smokeynet_adapted/data.py`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/smokeynet_adapted/data.py
git commit -m "feat(data): add load_frame_detections for sequence-level loading"
```

---

## Task 4: `select_longest_tube`

**Files:**
- Modify: `src/smokeynet_adapted/tubes.py`
- Modify: `tests/test_tubes.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tubes.py`:

```python
from smokeynet_adapted.tubes import select_longest_tube
from smokeynet_adapted.types import Detection, Tube, TubeEntry


def _det(cx: float = 0.5) -> Detection:
    return Detection(class_id=0, cx=cx, cy=0.5, w=0.1, h=0.1, confidence=1.0)


def _tube(tube_id: int, length: int, n_gap: int = 0) -> Tube:
    n_det = length - n_gap
    entries: list[TubeEntry] = []
    for i in range(n_det):
        entries.append(TubeEntry(frame_idx=i, detection=_det()))
    for i in range(n_gap):
        entries.append(
            TubeEntry(frame_idx=n_det + i, detection=None, is_gap=True)
        )
    return Tube(
        tube_id=tube_id,
        entries=entries,
        start_frame=0,
        end_frame=length - 1,
    )


def test_select_longest_tube_empty_returns_none():
    assert select_longest_tube([]) is None


def test_select_longest_tube_picks_longest():
    a = _tube(0, length=3)
    b = _tube(1, length=7)
    c = _tube(2, length=5)
    assert select_longest_tube([a, b, c]).tube_id == 1


def test_select_longest_tube_tie_break_by_non_gap_count():
    # Both span 5 frames, but a has 5 dets, b has only 3 (2 gaps).
    a = _tube(0, length=5, n_gap=0)
    b = _tube(1, length=5, n_gap=2)
    assert select_longest_tube([a, b]).tube_id == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tubes.py::test_select_longest_tube_empty_returns_none -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Edit `src/smokeynet_adapted/tubes.py`. Append:

```python
def select_longest_tube(tubes: list[Tube]) -> Tube | None:
    """Pick the single longest tube from a list.

    Length is measured as ``end_frame - start_frame + 1`` (so gaps count
    toward length). Ties are broken by the number of non-gap entries —
    the tube with more real observations wins. If still tied, the first
    in the input order wins.

    Args:
        tubes: Candidate tubes.

    Returns:
        The selected tube, or ``None`` if ``tubes`` is empty.
    """
    if not tubes:
        return None

    def _key(tube: Tube) -> tuple[int, int]:
        length = tube.end_frame - tube.start_frame + 1
        n_observed = sum(1 for e in tube.entries if e.detection is not None)
        return (length, n_observed)

    return max(tubes, key=_key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tubes.py -v -k "select_longest"`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/smokeynet_adapted/tubes.py tests/test_tubes.py
git commit -m "feat(tubes): add select_longest_tube with non-gap tie-break"
```

---

## Task 5: `interpolate_gaps`

**Files:**
- Modify: `src/smokeynet_adapted/tubes.py`
- Modify: `tests/test_tubes.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tubes.py`:

```python
from smokeynet_adapted.tubes import interpolate_gaps


def _entry(idx: int, det: Detection | None, is_gap: bool = False) -> TubeEntry:
    return TubeEntry(frame_idx=idx, detection=det, is_gap=is_gap)


def test_interpolate_gaps_interior_length_one_lerps_midpoint():
    before = Detection(0, cx=0.2, cy=0.3, w=0.1, h=0.1, confidence=0.9)
    after = Detection(0, cx=0.4, cy=0.5, w=0.2, h=0.2, confidence=0.7)
    tube = Tube(
        tube_id=0,
        entries=[
            _entry(0, before),
            _entry(1, None, is_gap=True),
            _entry(2, after),
        ],
        start_frame=0,
        end_frame=2,
    )
    out = interpolate_gaps(tube)
    gap = out.entries[1]
    assert gap.is_gap is True
    assert gap.detection is not None
    assert gap.detection.cx == pytest.approx(0.3)
    assert gap.detection.cy == pytest.approx(0.4)
    assert gap.detection.w == pytest.approx(0.15)
    assert gap.detection.h == pytest.approx(0.15)
    assert gap.detection.confidence == pytest.approx(0.0)


def test_interpolate_gaps_leading_gap_repeats_first_observed():
    first = Detection(0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.8)
    tube = Tube(
        tube_id=0,
        entries=[
            _entry(0, None, is_gap=True),
            _entry(1, first),
        ],
        start_frame=0,
        end_frame=1,
    )
    out = interpolate_gaps(tube)
    g = out.entries[0]
    assert g.is_gap is True
    assert g.detection is not None
    assert g.detection.cx == pytest.approx(0.5)
    assert g.detection.cy == pytest.approx(0.5)
    assert g.detection.confidence == pytest.approx(0.0)


def test_interpolate_gaps_trailing_gap_repeats_last_observed():
    last = Detection(0, cx=0.7, cy=0.2, w=0.05, h=0.05, confidence=0.9)
    tube = Tube(
        tube_id=0,
        entries=[
            _entry(0, last),
            _entry(1, None, is_gap=True),
        ],
        start_frame=0,
        end_frame=1,
    )
    out = interpolate_gaps(tube)
    g = out.entries[1]
    assert g.is_gap is True
    assert g.detection is not None
    assert g.detection.cx == pytest.approx(0.7)
    assert g.detection.confidence == pytest.approx(0.0)


def test_interpolate_gaps_observed_entries_unchanged():
    obs = Detection(0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.9)
    tube = Tube(
        tube_id=0,
        entries=[_entry(0, obs)],
        start_frame=0,
        end_frame=0,
    )
    out = interpolate_gaps(tube)
    assert out.entries[0].is_gap is False
    assert out.entries[0].detection is obs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tubes.py -v -k "interpolate_gaps"`
Expected: ImportError on `interpolate_gaps`.

- [ ] **Step 3: Implement**

Edit `src/smokeynet_adapted/tubes.py`. Append:

```python
def interpolate_gaps(tube: Tube) -> Tube:
    """Fill gap entries with a geometrically-interpolated bbox.

    For each entry whose ``detection`` is ``None``:

    * **Interior gap** (observed dets on both sides): linearly interpolate
      ``(cx, cy, w, h)`` between the nearest observed detection before and
      after, using the entry's ``frame_idx`` as the position parameter.
    * **Boundary gap** (no observation on one side): repeat the nearest
      observed detection on the other side.

    Synthesized detections always carry ``confidence=0.0``. The returned
    tube has ``is_gap=True`` flags on every previously-empty entry.

    Observed entries are left untouched.

    Args:
        tube: Tube whose gap entries (``detection=None``) need filling.

    Returns:
        The same tube object, mutated in place. Returned for chaining.
    """
    observed = [
        (i, e.detection)
        for i, e in enumerate(tube.entries)
        if e.detection is not None
    ]
    if not observed:
        return tube

    for i, entry in enumerate(tube.entries):
        if entry.detection is not None:
            continue

        before = next(
            ((j, d) for j, d in reversed(observed) if j < i),
            None,
        )
        after = next(
            ((j, d) for j, d in observed if j > i),
            None,
        )

        if before is not None and after is not None:
            j_b, d_b = before
            j_a, d_a = after
            t = (i - j_b) / (j_a - j_b)
            cx = d_b.cx + t * (d_a.cx - d_b.cx)
            cy = d_b.cy + t * (d_a.cy - d_b.cy)
            w = d_b.w + t * (d_a.w - d_b.w)
            h = d_b.h + t * (d_a.h - d_b.h)
            class_id = d_b.class_id
        elif before is not None:
            d = before[1]
            cx, cy, w, h, class_id = d.cx, d.cy, d.w, d.h, d.class_id
        else:
            d = after[1]
            cx, cy, w, h, class_id = d.cx, d.cy, d.w, d.h, d.class_id

        entry.detection = Detection(
            class_id=class_id,
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            confidence=0.0,
        )
        entry.is_gap = True

    return tube
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tubes.py -v -k "interpolate_gaps"`
Expected: all 4 PASS.

- [ ] **Step 5: Run the full tube test file to make sure nothing regressed**

Run: `uv run pytest tests/test_tubes.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/smokeynet_adapted/tubes.py tests/test_tubes.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/smokeynet_adapted/tubes.py tests/test_tubes.py
git commit -m "feat(tubes): geometric gap interpolation with boundary repeat"
```

---

## Task 6: `scripts/build_tubes.py`

**Files:**
- Create: `scripts/build_tubes.py`

No unit test — the notebook (Task 9) and the DVC stage (Task 7) act as the smoke test.

- [ ] **Step 1: Implement**

Create `scripts/build_tubes.py`:

```python
"""Build per-sequence smoke tube JSON dataset from label .txt files.

For each sequence under ``--input-dir/{wildfire,fp}/``:

1. Load detections from labels (5-col GT for wildfire, 6-col YOLO for fp).
2. Build candidate tubes via greedy IoU matching.
3. Select the longest tube; tie-break by non-gap entries.
4. Geometrically interpolate gap bboxes.
5. Apply length / observation filters.
6. Write a JSON file per surviving sequence and a summary file with
   per-split stats and dropped-sequence reasons.

No YOLO inference is performed — the labels carry everything we need.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from smokeynet_adapted.data import (
    is_wf_sequence,
    list_sequences,
    load_frame_detections,
)
from smokeynet_adapted.tubes import (
    build_tubes,
    interpolate_gaps,
    select_longest_tube,
)
from smokeynet_adapted.types import Tube


@dataclass
class DropRecord:
    sequence_id: str
    reason: str


def _serialize_tube(
    *,
    sequence_id: str,
    split: str,
    label: str,
    source: str,
    num_frames: int,
    tube: Tube,
    frame_id_by_idx: dict[int, str],
) -> dict:
    return {
        "sequence_id": sequence_id,
        "split": split,
        "label": label,
        "source": source,
        "num_frames": num_frames,
        "tube": {
            "start_frame": tube.start_frame,
            "end_frame": tube.end_frame,
            "entries": [
                {
                    "frame_idx": e.frame_idx,
                    "frame_id": frame_id_by_idx.get(e.frame_idx, ""),
                    "bbox": [
                        e.detection.cx,
                        e.detection.cy,
                        e.detection.w,
                        e.detection.h,
                    ]
                    if e.detection is not None
                    else None,
                    "is_gap": e.is_gap,
                    "confidence": e.detection.confidence
                    if e.detection is not None
                    else None,
                }
                for e in tube.entries
            ],
        },
    }


def _process_sequence(
    seq_dir: Path,
    *,
    split: str,
    iou_threshold: float,
    max_misses: int,
    min_tube_length: int,
    min_detected_entries: int,
) -> tuple[dict | None, str | None]:
    """Process a single sequence.

    Returns ``(record_or_None, drop_reason_or_None)``.
    """
    is_wf = is_wf_sequence(seq_dir)
    label = "smoke" if is_wf else "fp"
    source = "gt" if is_wf else "yolo"

    if not (seq_dir / "labels").is_dir():
        return None, "no_labels_dir"

    fdets = load_frame_detections(seq_dir)
    if not fdets:
        return None, "no_frames"

    total_dets = sum(len(fd.detections) for fd in fdets)
    if total_dets < min_detected_entries:
        return None, "no_detections"

    tubes = build_tubes(fdets, iou_threshold=iou_threshold, max_misses=max_misses)
    if not tubes:
        return None, "no_tubes"

    selected = select_longest_tube(tubes)
    assert selected is not None  # tubes is non-empty

    length = selected.end_frame - selected.start_frame + 1
    if length < min_tube_length:
        return None, "too_short"

    n_observed = sum(1 for e in selected.entries if e.detection is not None)
    if n_observed < min_detected_entries:
        return None, "too_few_observed"

    interpolate_gaps(selected)

    frame_id_by_idx = {fd.frame_idx: fd.frame_id for fd in fdets}
    record = _serialize_tube(
        sequence_id=seq_dir.name,
        split=split,
        label=label,
        source=source,
        num_frames=len(fdets),
        tube=selected,
        frame_id_by_idx=frame_id_by_idx,
    )
    return record, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--max-misses", type=int, default=2)
    parser.add_argument("--min-tube-length", type=int, default=4)
    parser.add_argument("--min-detected-entries", type=int, default=2)
    args = parser.parse_args()

    split = args.input_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = list_sequences(args.input_dir)
    written = 0
    by_label: dict[str, int] = {"smoke": 0, "fp": 0}
    dropped: list[DropRecord] = []

    for seq_dir in seq_dirs:
        record, reason = _process_sequence(
            seq_dir,
            split=split,
            iou_threshold=args.iou_threshold,
            max_misses=args.max_misses,
            min_tube_length=args.min_tube_length,
            min_detected_entries=args.min_detected_entries,
        )
        if reason is not None:
            dropped.append(DropRecord(sequence_id=seq_dir.name, reason=reason))
            continue

        out_path = args.output_dir / f"{seq_dir.name}.json"
        out_path.write_text(json.dumps(record, indent=2))
        written += 1
        by_label[record["label"]] += 1

    summary = {
        "split": split,
        "total_sequences": len(seq_dirs),
        "tubes_written": written,
        "by_label": by_label,
        "dropped": [{"sequence_id": d.sequence_id, "reason": d.reason} for d in dropped],
    }
    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"[{split}] wrote {written}/{len(seq_dirs)} tubes "
        f"(smoke={by_label['smoke']}, fp={by_label['fp']}, "
        f"dropped={len(dropped)})"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check scripts/build_tubes.py`
Expected: no errors.

- [ ] **Step 3: Smoke run on val (smaller)**

Run:
```
uv run python scripts/build_tubes.py \
  --input-dir data/01_raw/datasets/val \
  --output-dir /tmp/tubes_val_smoketest
```

Expected: prints `[val] wrote N/M tubes (smoke=..., fp=..., dropped=...)`. N should be roughly 90%+ of M. Inspect a couple of files:

```
ls /tmp/tubes_val_smoketest | head -5
cat /tmp/tubes_val_smoketest/_summary.json | head -40
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_tubes.py
git commit -m "feat(scripts): build per-sequence smoke tube JSON dataset"
```

---

## Task 7: Wire DVC stage + params

**Files:**
- Modify: `params.yaml`
- Modify: `dvc.yaml`

- [ ] **Step 1: Add `build_tubes` params**

Edit `params.yaml`. After the existing `tubes:` block (around line 18-20), append:

```yaml
build_tubes:
  min_tube_length: 4
  min_detected_entries: 2
```

- [ ] **Step 2: Add the DVC stage**

Edit `dvc.yaml`. After the `truncate` stage (around line 12) and before the commented-out `prepare:` line, insert:

```yaml
  build_tubes:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/build_tubes.py
        --input-dir data/01_raw/datasets/${item}
        --output-dir data/03_primary/tubes/${item}
        --iou-threshold ${tubes.iou_threshold}
        --max-misses ${tubes.max_misses}
        --min-tube-length ${build_tubes.min_tube_length}
        --min-detected-entries ${build_tubes.min_detected_entries}
      deps:
        - scripts/build_tubes.py
        - src/smokeynet_adapted/tubes.py
        - src/smokeynet_adapted/data.py
        - src/smokeynet_adapted/types.py
        - data/01_raw/datasets/${item}
      params:
        - tubes
        - build_tubes
      outs:
        - data/03_primary/tubes/${item}
```

- [ ] **Step 3: Reproduce the stage for both splits**

Run:
```
uv run dvc repro build_tubes@val
uv run dvc repro build_tubes@train
```

Expected: both succeed; `data/03_primary/tubes/{train,val}/` populated with `<sequence_id>.json` files plus `_summary.json`. Check:

```
ls data/03_primary/tubes/val | wc -l        # ~150 + 1 summary
cat data/03_primary/tubes/val/_summary.json | python -m json.tool | head -20
ls data/03_primary/tubes/train | wc -l      # ~1500 + 1 summary
```

If the WF dropout rate exceeds ~10% (visible in `_summary.json`), revisit thresholds in `params.yaml` (loosen `min_tube_length` to 3 and re-run).

- [ ] **Step 4: Commit**

```bash
git add params.yaml dvc.yaml dvc.lock data/03_primary/tubes/train.dvc data/03_primary/tubes/val.dvc
git commit -m "feat(dvc): add build_tubes stage"
```

If DVC produced different output paths (e.g., one `.dvc` file per directory under another path), `git status` will show the truth — add whatever it actually created.

---

## Task 8: Visualization notebook

**Files:**
- Create: `notebooks/02-visualize-built-tubes.ipynb`

- [ ] **Step 1: Create the notebook**

Create the file via `uv run jupyter`:

```bash
uv run jupyter nbconvert --to notebook --output notebooks/02-visualize-built-tubes.ipynb /dev/null 2>/dev/null || true
```

If that doesn't work, write the file directly. Easiest: use the Write tool to create the file with the JSON content below (a minimal valid notebook).

Content of `notebooks/02-visualize-built-tubes.ipynb`:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Visualize built smoke tubes\n", "\n", "Loads tube JSONs from `data/03_primary/tubes/{split}/` and renders timelines + filmstrips. No YOLO model required."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from smokeynet_adapted.data import find_sequence_dir, get_sorted_frames\n",
    "from smokeynet_adapted.tubes import plot_tube_summary\n",
    "from smokeynet_adapted.types import Detection, Tube, TubeEntry"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "SPLIT = \"val\"\n",
    "TUBES_DIR = Path(f\"../data/03_primary/tubes/{SPLIT}\")\n",
    "RAW_DIR = Path(f\"../data/01_raw/datasets/{SPLIT}\")\n",
    "NUM_TO_SHOW = 6\n",
    "\n",
    "summary = json.loads((TUBES_DIR / \"_summary.json\").read_text())\n",
    "print(f\"split={summary['split']}  written={summary['tubes_written']}/{summary['total_sequences']}  by_label={summary['by_label']}  dropped={len(summary['dropped'])}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def _record_to_tube(record: dict) -> Tube:\n",
    "    t = record[\"tube\"]\n",
    "    entries: list[TubeEntry] = []\n",
    "    for e in t[\"entries\"]:\n",
    "        bbox = e[\"bbox\"]\n",
    "        det = (\n",
    "            Detection(\n",
    "                class_id=0,\n",
    "                cx=bbox[0],\n",
    "                cy=bbox[1],\n",
    "                w=bbox[2],\n",
    "                h=bbox[3],\n",
    "                confidence=e[\"confidence\"] or 0.0,\n",
    "            )\n",
    "            if bbox is not None\n",
    "            else None\n",
    "        )\n",
    "        entries.append(\n",
    "            TubeEntry(frame_idx=e[\"frame_idx\"], detection=det, is_gap=e[\"is_gap\"])\n",
    "        )\n",
    "    return Tube(\n",
    "        tube_id=0,\n",
    "        entries=entries,\n",
    "        start_frame=t[\"start_frame\"],\n",
    "        end_frame=t[\"end_frame\"],\n",
    "    )\n",
    "\n",
    "tube_files = sorted(p for p in TUBES_DIR.glob(\"*.json\") if p.name != \"_summary.json\")\n",
    "print(f\"{len(tube_files)} tube files\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for path in tube_files[:NUM_TO_SHOW]:\n",
    "    record = json.loads(path.read_text())\n",
    "    tube = _record_to_tube(record)\n",
    "    seq_dir = find_sequence_dir(RAW_DIR, record[\"sequence_id\"])\n",
    "    if seq_dir is None:\n",
    "        print(f\"skipping {record['sequence_id']}: source dir not found\")\n",
    "        continue\n",
    "    frame_paths = get_sorted_frames(seq_dir)\n",
    "    fig = plot_tube_summary(\n",
    "        frame_paths,\n",
    "        [tube],\n",
    "        num_frames=record[\"num_frames\"],\n",
    "        tube_labels=[record[\"label\"] == \"smoke\"],\n",
    "        title=f\"{record['sequence_id']} [{record['label']}]\",\n",
    "    )\n",
    "    plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Lint the notebook (nbqa)**

Run: `uv run nbqa ruff notebooks/02-visualize-built-tubes.ipynb`
Expected: no errors. Fix imports/whitespace if any are flagged.

- [ ] **Step 3: Run the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/02-visualize-built-tubes.ipynb --output 02-visualize-built-tubes.ipynb`
Expected: completes without exceptions; produces 6 tube visualizations from the val split. After execution, run `uv run nbstripout notebooks/02-visualize-built-tubes.ipynb` to strip outputs (matches the project's `nbstripout` convention — `make install` configures the hook, but stripping manually is safe).

- [ ] **Step 4: Commit**

```bash
git add notebooks/02-visualize-built-tubes.ipynb
git commit -m "feat(notebooks): visualize built smoke tube dataset"
```

---

## Task 9: Final verification

- [ ] **Step 1: Lint everything**

Run: `uv run make lint`
Expected: no errors.

- [ ] **Step 2: Run full test suite**

Run: `uv run make test`
Expected: all PASS — including the 4 new `load_detections` tests, the 3 new `select_longest_tube` tests, and the 4 new `interpolate_gaps` tests.

- [ ] **Step 3: Confirm DVC outputs are reproducible**

Run: `uv run dvc status`
Expected: `build_tubes@train` and `build_tubes@val` both report up-to-date.

- [ ] **Step 4: Eyeball summary stats**

Run:
```
cat data/03_primary/tubes/train/_summary.json | python -c "import json,sys; s=json.load(sys.stdin); print('train', s['tubes_written'], '/', s['total_sequences'], s['by_label'])"
cat data/03_primary/tubes/val/_summary.json   | python -c "import json,sys; s=json.load(sys.stdin); print('val  ', s['tubes_written'], '/', s['total_sequences'], s['by_label'])"
```

Expected: ratios in line with the spec's pre-spike measurement (~94% survival on FP). If WF survival is markedly worse, revisit `min_tube_length` in `params.yaml`.

---

## Spec coverage check

| Spec section | Covered by |
|---|---|
| `TubeEntry` extension | Task 1 |
| `load_detections` (5/6-col) | Task 2 |
| `load_frame_detections` | Task 3 |
| `select_longest_tube` | Task 4 |
| `interpolate_gaps` (lerp + boundary repeat, conf=0.0) | Task 5 |
| `scripts/build_tubes.py` (CLI, `uv run`) | Task 6 |
| `params.yaml` `build_tubes` section | Task 7 |
| `dvc.yaml` `build_tubes` foreach stage | Task 7 |
| `notebooks/02-visualize-built-tubes.ipynb` | Task 8 |
| Filters (min_tube_length=4, min_detected_entries=2) | Tasks 6 & 7 |
| Drop reasons in `_summary.json` | Task 6 |
| Output schema (`label`, `source`, `bbox`, `is_gap`, `confidence`) | Task 6 |
| Test plan (interpolate_gaps + select_longest_tube + load_detections) | Tasks 2, 4, 5 |
| Verification (`make lint`, `make test`, `dvc repro`, notebook smoke test) | Task 9 |

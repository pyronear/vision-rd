# Render Tubes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render one PNG per built smoke tube to `data/08_reporting/tubes/{train,val}/{smoke,fp}/<seq>.png` so the dataset can be reviewed quickly with a file explorer.

**Architecture:** New `scripts/render_tubes.py` walks tube JSONs from `data/03_primary/tubes/<split>/`, reuses the existing `plot_tube_summary` to render each tube as a PNG into a label-nested output tree. Two small refactors first: extract `_record_to_tube` from the notebook into `tube_from_record` in `tubes.py`, add a `load_tube_record` helper to `data.py`, and have the notebook use them. A new `render_tubes@{train,val}` DVC stage wires it into the pipeline.

**Tech Stack:** Python 3.11+, uv, matplotlib, pytest, ruff, DVC. All commands run with `uv run`. Spec: `docs/specs/2026-04-14-render-tubes-design.md`.

**User commit preference:** Never include `Co-Authored-By: Claude` or any mention of Claude/Anthropic in commit messages.

**Working directory for ALL commands below:** `experiments/temporal-models/smokeynet-adapted/`

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `src/smokeynet_adapted/data.py` | Modify | Add `load_tube_record(path) -> dict` |
| `src/smokeynet_adapted/tubes.py` | Modify | Add `tube_from_record(record) -> Tube` |
| `tests/test_data.py` | Modify | Round-trip test for `load_tube_record` |
| `tests/test_tubes.py` | Modify | Round-trip test for `tube_from_record` |
| `notebooks/02-visualize-built-tubes.ipynb` | Modify | Replace inline `_record_to_tube` with imports |
| `scripts/render_tubes.py` (new) | Create | CLI: walk JSONs → render PNGs |
| `dvc.yaml` | Modify | Add `render_tubes@{train,val}` stage |

---

## Task 1: `load_tube_record` in `data.py`

**Files:**
- Modify: `src/smokeynet_adapted/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data.py`:

```python
import json

from smokeynet_adapted.data import load_tube_record


def test_load_tube_record_roundtrip(tmp_path):
    path = tmp_path / "seq.json"
    record = {
        "sequence_id": "abc",
        "split": "val",
        "label": "smoke",
        "source": "gt",
        "num_frames": 3,
        "tube": {
            "start_frame": 0,
            "end_frame": 2,
            "entries": [
                {
                    "frame_idx": 0,
                    "frame_id": "f0",
                    "bbox": [0.5, 0.5, 0.1, 0.1],
                    "is_gap": False,
                    "confidence": 1.0,
                }
            ],
        },
    }
    path.write_text(json.dumps(record))
    assert load_tube_record(path) == record
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py::test_load_tube_record_roundtrip -v`
Expected: FAIL with `ImportError: cannot import name 'load_tube_record'`.

- [ ] **Step 3: Implement**

Add this import near the top of `src/smokeynet_adapted/data.py` (after the existing `import re`):

```python
import json
```

Then append at the end of the file:

```python
def load_tube_record(path: Path) -> dict:
    """Read+parse a tube JSON file.

    Trivial wrapper around :func:`json.loads`; exists so callers
    (scripts, notebooks) have a single named entry point for tube I/O.

    Args:
        path: Path to a tube ``.json`` file produced by
            ``scripts/build_tubes.py``.

    Returns:
        The parsed record as a plain dict.
    """
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: all PASS, including the new round-trip test.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/smokeynet_adapted/data.py tests/test_data.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/smokeynet_adapted/data.py tests/test_data.py
git commit -m "feat(data): add load_tube_record helper for tube JSON I/O"
```

---

## Task 2: `tube_from_record` in `tubes.py`

**Files:**
- Modify: `src/smokeynet_adapted/tubes.py`
- Modify: `tests/test_tubes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tubes.py` (the imports already include `Tube`, `TubeEntry`, `Detection`):

```python
from smokeynet_adapted.tubes import tube_from_record


class TestTubeFromRecord:
    def test_rebuilds_observed_entry(self):
        record = {
            "tube": {
                "start_frame": 0,
                "end_frame": 0,
                "entries": [
                    {
                        "frame_idx": 0,
                        "frame_id": "f0",
                        "bbox": [0.4, 0.5, 0.2, 0.3],
                        "is_gap": False,
                        "confidence": 0.9,
                    }
                ],
            }
        }
        tube = tube_from_record(record)
        assert tube.start_frame == 0
        assert tube.end_frame == 0
        assert len(tube.entries) == 1
        e = tube.entries[0]
        assert e.frame_idx == 0
        assert e.is_gap is False
        assert e.detection is not None
        assert e.detection.cx == 0.4
        assert e.detection.cy == 0.5
        assert e.detection.w == 0.2
        assert e.detection.h == 0.3
        assert e.detection.confidence == 0.9

    def test_rebuilds_gap_entry(self):
        record = {
            "tube": {
                "start_frame": 0,
                "end_frame": 0,
                "entries": [
                    {
                        "frame_idx": 0,
                        "frame_id": "f0",
                        "bbox": [0.5, 0.5, 0.1, 0.1],
                        "is_gap": True,
                        "confidence": 0.0,
                    }
                ],
            }
        }
        tube = tube_from_record(record)
        e = tube.entries[0]
        assert e.is_gap is True
        assert e.detection is not None
        assert e.detection.confidence == 0.0

    def test_rebuilds_missing_bbox_as_none_detection(self):
        record = {
            "tube": {
                "start_frame": 0,
                "end_frame": 0,
                "entries": [
                    {
                        "frame_idx": 0,
                        "frame_id": "f0",
                        "bbox": None,
                        "is_gap": True,
                        "confidence": None,
                    }
                ],
            }
        }
        tube = tube_from_record(record)
        e = tube.entries[0]
        assert e.detection is None
        assert e.is_gap is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tubes.py::TestTubeFromRecord -v`
Expected: FAIL with `ImportError: cannot import name 'tube_from_record'`.

- [ ] **Step 3: Implement**

Append at the end of `src/smokeynet_adapted/tubes.py`:

```python
def tube_from_record(record: dict) -> Tube:
    """Rebuild a :class:`Tube` from a tube JSON record.

    Inverse of ``_serialize_tube`` in ``scripts/build_tubes.py``. Pure
    function; no I/O.

    Entries with ``bbox=None`` are reconstructed with ``detection=None``
    (pre-interpolation gap shape). Otherwise a :class:`Detection` is
    built from the bbox + confidence; ``confidence=None`` falls back to
    ``0.0``.

    Args:
        record: Parsed tube record. Only the ``tube`` sub-object is
            consulted; other top-level keys are ignored.

    Returns:
        A :class:`Tube` with ``tube_id=0`` (the on-disk dataset is
        single-tube-per-sequence so the id is informational only).
    """
    t = record["tube"]
    entries: list[TubeEntry] = []
    for e in t["entries"]:
        bbox = e["bbox"]
        if bbox is None:
            det: Detection | None = None
        else:
            det = Detection(
                class_id=0,
                cx=bbox[0],
                cy=bbox[1],
                w=bbox[2],
                h=bbox[3],
                confidence=e["confidence"] if e["confidence"] is not None else 0.0,
            )
        entries.append(
            TubeEntry(
                frame_idx=e["frame_idx"],
                detection=det,
                is_gap=e["is_gap"],
            )
        )
    return Tube(
        tube_id=0,
        entries=entries,
        start_frame=t["start_frame"],
        end_frame=t["end_frame"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tubes.py -v`
Expected: all PASS, including the 3 new `TestTubeFromRecord` tests.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/smokeynet_adapted/tubes.py tests/test_tubes.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/smokeynet_adapted/tubes.py tests/test_tubes.py
git commit -m "feat(tubes): add tube_from_record to rebuild Tube from JSON dict"
```

---

## Task 3: Switch the notebook to the shared helpers

**Files:**
- Modify: `notebooks/02-visualize-built-tubes.ipynb`

The notebook currently has an inline `_record_to_tube` and uses `json.loads(path.read_text())` directly. Replace with the new shared helpers.

- [ ] **Step 1: Replace the imports cell**

Use the `NotebookEdit` tool to replace the cell with `id="imports"` (cell index 1) with this source:

```python
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from smokeynet_adapted.data import find_sequence_dir, get_sorted_frames, load_tube_record
from smokeynet_adapted.tubes import plot_tube_summary, tube_from_record
```

(The `defaultdict` import is unused; if `nbqa ruff` flags it, drop the line. The point of this step is to import the shared helpers.)

- [ ] **Step 2: Replace the loader cell**

Replace cell `id="loader"` (cell index 3) with this source:

```python
all_files = sorted(p for p in TUBES_DIR.glob("*.json") if p.name != "_summary.json")
by_label: dict[str, list[Path]] = {"smoke": [], "fp": []}
for p in all_files:
    rec = load_tube_record(p)
    by_label[rec["label"]].append(p)
print(f"smoke files: {len(by_label['smoke'])}, fp files: {len(by_label['fp'])}")

selected = by_label["smoke"][:PER_LABEL] + by_label["fp"][:PER_LABEL]
print(f"showing {len(selected)} sequences ({PER_LABEL} smoke + {PER_LABEL} fp)")
```

(This drops the inline `_record_to_tube` definition entirely.)

- [ ] **Step 3: Replace the render cell**

Replace cell `id="render"` (cell index 4) with this source:

```python
for path in selected:
    record = load_tube_record(path)
    tube = tube_from_record(record)
    seq_dir = find_sequence_dir(RAW_DIR, record["sequence_id"])
    if seq_dir is None:
        print(f"skipping {record['sequence_id']}: source dir not found")
        continue
    frame_paths = get_sorted_frames(seq_dir)
    fig = plot_tube_summary(
        frame_paths,
        [tube],
        num_frames=record["num_frames"],
        tube_labels=[record["label"] == "smoke"],
        title=f"{record['sequence_id']} [{record['label']}]",
    )
    plt.show()
```

- [ ] **Step 4: Lint the notebook**

Run: `uv run nbqa ruff notebooks/02-visualize-built-tubes.ipynb`
Expected: no errors. If `defaultdict` is flagged as unused, edit the imports cell again to drop that line.

- [ ] **Step 5: Execute end-to-end**

Run:
```
uv run jupyter nbconvert --to notebook --execute notebooks/02-visualize-built-tubes.ipynb --output 02-visualize-built-tubes.ipynb
uv run nbstripout notebooks/02-visualize-built-tubes.ipynb
```
Expected: notebook runs without exceptions and produces 6 figures (3 smoke + 3 fp).

- [ ] **Step 6: Commit**

```bash
git add notebooks/02-visualize-built-tubes.ipynb
git commit -m "refactor(notebooks): use shared load_tube_record + tube_from_record"
```

---

## Task 4: `scripts/render_tubes.py`

**Files:**
- Create: `scripts/render_tubes.py`

No unit test — verified by running it on the val split and inspecting the PNGs.

- [ ] **Step 1: Implement**

Create `scripts/render_tubes.py`:

```python
"""Render one PNG per built smoke tube to a label-nested directory tree.

For each tube JSON in ``--tubes-dir``, look up the source sequence
under ``--raw-dir``, render ``plot_tube_summary``, and save the PNG to
``--output-dir/{smoke,fp}/<sequence_id>.png``.

The output directory is wiped at the start of each run so stale PNGs
from earlier runs (for sequences that have since been filtered out)
don't linger.
"""

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from smokeynet_adapted.data import (
    find_sequence_dir,
    get_sorted_frames,
    load_tube_record,
)
from smokeynet_adapted.tubes import plot_tube_summary, tube_from_record


def _render_one(
    record_path: Path,
    *,
    raw_dir: Path,
    output_dir: Path,
    dpi: int,
) -> tuple[str | None, str | None]:
    """Render a single tube to PNG.

    Returns ``(label, None)`` on success, ``(None, reason)`` on skip.
    """
    record = load_tube_record(record_path)
    sequence_id = record["sequence_id"]
    label = record["label"]
    seq_dir = find_sequence_dir(raw_dir, sequence_id)
    if seq_dir is None:
        return None, f"source dir not found for {sequence_id}"

    frame_paths = get_sorted_frames(seq_dir)
    if not frame_paths:
        return None, f"no frames for {sequence_id}"

    tube = tube_from_record(record)

    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_dir / f"{sequence_id}.png"

    fig = plot_tube_summary(
        frame_paths,
        [tube],
        num_frames=record["num_frames"],
        tube_labels=[label == "smoke"],
        title=f"{sequence_id} [{label}]",
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return label, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tubes-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    split = args.tubes_dir.name

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record_paths = sorted(
        p for p in args.tubes_dir.glob("*.json") if p.name != "_summary.json"
    )

    if not record_paths:
        print(f"[{split}] no tube records found in {args.tubes_dir}; nothing to render")
        return

    by_label: dict[str, int] = {"smoke": 0, "fp": 0}
    skipped: list[str] = []

    for path in record_paths:
        label, reason = _render_one(
            path,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
        if reason is not None:
            skipped.append(reason)
            continue
        assert label is not None
        by_label[label] += 1

    rendered = sum(by_label.values())
    print(
        f"[{split}] rendered {rendered}/{len(record_paths)} tubes "
        f"(smoke={by_label['smoke']}, fp={by_label['fp']}, skipped={len(skipped)})"
    )
    for reason in skipped[:5]:
        print(f"  skipped: {reason}")
    if len(skipped) > 5:
        print(f"  ... and {len(skipped) - 5} more")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check scripts/render_tubes.py`
Expected: no errors.

- [ ] **Step 3: Smoke run on val**

Run:
```
uv run python scripts/render_tubes.py \
  --tubes-dir data/03_primary/tubes/val \
  --raw-dir data/01_raw/datasets/val \
  --output-dir /tmp/render_tubes_smoketest
```

Expected: prints `[val] rendered N/M tubes (smoke=A, fp=B, skipped=K)` where N is close to M (~280).

Inspect a couple of PNGs:
```
ls /tmp/render_tubes_smoketest/smoke | head -3
ls /tmp/render_tubes_smoketest/fp | head -3
file /tmp/render_tubes_smoketest/smoke/$(ls /tmp/render_tubes_smoketest/smoke | head -1)
```

Each file should be a valid PNG image. Open one or two visually if you want to eyeball the rendering.

- [ ] **Step 4: Cleanup smoke-test output and commit**

```bash
rm -rf /tmp/render_tubes_smoketest
git add scripts/render_tubes.py
git commit -m "feat(scripts): render tube PNGs to label-nested report tree"
```

---

## Task 5: Wire DVC stage

**Files:**
- Modify: `dvc.yaml`

- [ ] **Step 1: Add the `render_tubes` stage**

Edit `dvc.yaml`. After the `build_tubes` stage (the one with `outs: data/03_primary/tubes/${item}`) and before the next commented-out stage, add:

```yaml
  render_tubes:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/render_tubes.py
        --tubes-dir data/03_primary/tubes/${item}
        --raw-dir data/01_raw/datasets/${item}
        --output-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/render_tubes.py
        - src/smokeynet_adapted/tubes.py
        - src/smokeynet_adapted/data.py
        - data/03_primary/tubes/${item}
        - data/01_raw/datasets/${item}
      outs:
        - data/08_reporting/tubes/${item}
```

- [ ] **Step 2: Reproduce both splits**

Run:
```
uv run dvc repro render_tubes@val
uv run dvc repro render_tubes@train
```

Expected: both succeed; populate `data/08_reporting/tubes/{train,val}/{smoke,fp}/`. Check:

```
ls data/08_reporting/tubes/val/smoke | wc -l   # ~135
ls data/08_reporting/tubes/val/fp | wc -l      # ~149
ls data/08_reporting/tubes/train/smoke | wc -l # ~1399
ls data/08_reporting/tubes/train/fp | wc -l    # ~1443
```

- [ ] **Step 3: Verify file-explorer browsability**

Open `data/08_reporting/tubes/val/smoke/` in any image viewer or file explorer with thumbnails. PNGs should render at thumbnail size and clearly show the smoke tube timeline + filmstrip. Repeat for `fp/`.

- [ ] **Step 4: Commit**

Run `git status --short` to see exactly what DVC produced (likely `dvc.lock` plus a `data/08_reporting/tubes/.gitignore`). Stage and commit only those files plus `dvc.yaml`:

```bash
git add dvc.yaml dvc.lock
# Add the .gitignore DVC produced (path may vary slightly):
git add data/08_reporting/tubes/.gitignore
git commit -m "feat(dvc): add render_tubes stage for browsable PNG reports"
```

If DVC produced different output paths, add what `git status` actually shows.

---

## Task 6: Final verification

- [ ] **Step 1: Lint everything**

Run: `uv run make lint`
Expected: no errors.

- [ ] **Step 2: Run full test suite**

Run: `uv run make test`
Expected: all PASS — including the new `load_tube_record` round-trip test and the 3 `TestTubeFromRecord` tests.

- [ ] **Step 3: Confirm DVC is clean**

Run: `uv run dvc status`
Expected: `Data and pipelines are up to date.`

- [ ] **Step 4: Eyeball output counts**

Run:
```
echo "val:   smoke=$(ls data/08_reporting/tubes/val/smoke 2>/dev/null | wc -l)  fp=$(ls data/08_reporting/tubes/val/fp 2>/dev/null | wc -l)"
echo "train: smoke=$(ls data/08_reporting/tubes/train/smoke 2>/dev/null | wc -l)  fp=$(ls data/08_reporting/tubes/train/fp 2>/dev/null | wc -l)"
```

Expected: counts match the per-label totals from `_summary.json` for each split.

---

## Spec coverage check

| Spec section | Covered by |
|---|---|
| `load_tube_record` in `data.py` | Task 1 |
| `tube_from_record` in `tubes.py` | Task 2 |
| Notebook uses shared helpers | Task 3 |
| `scripts/render_tubes.py` (CLI, `uv run`) | Task 4 |
| Wipe output dir at script start | Task 4 |
| Skip-on-missing-source-dir | Task 4 |
| Empty input dir → exit 0 + warn | Task 4 |
| `dvc.yaml` `render_tubes` foreach stage | Task 5 |
| Output layout `08_reporting/tubes/<split>/<label>/<seq>.png` | Task 4 (mkdir) + Task 5 (DVC) |
| Tests: `load_tube_record` + `tube_from_record` | Tasks 1, 2 |
| Verification (`make lint`, `make test`, `dvc status`) | Task 6 |

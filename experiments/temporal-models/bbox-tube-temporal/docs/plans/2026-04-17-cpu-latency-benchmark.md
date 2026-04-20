# CPU latency benchmark — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/benchmark_cpu_latency.py`, a standalone script that measures per-sequence latency of a packaged bbox-tube-temporal model on CPU (or any chosen device), split by YOLO vs. classifier, with warmup discard and JSON + terminal output.

**Architecture:** One script, one test file. The script exposes a pure `run_benchmark_on_model(model, sequence_dirs, warmup)` function plus a CLI wrapper. Timing is non-invasive: two lightweight proxies (`TimedYoloProxy`, `TimedClassifier`) wrap the packaged model's `_yolo` and `_classifier` attributes immediately after `from_archive`, accumulating into a shared bucket dict that the driver zeros between sequences.

**Tech Stack:** Python 3.11+, PyTorch, Pydantic (already a dep), tqdm, argparse, pytest. All commands run from `experiments/temporal-models/bbox-tube-temporal/`.

**Spec:** `docs/specs/2026-04-17-cpu-latency-benchmark-design.md`.

**Commit style:** no Claude / Anthropic co-author trailers. Stage files explicitly, never `git add -A`.

---

## File Structure

- **Create** `scripts/benchmark_cpu_latency.py` — single-file script with:
  - `percentile(xs, p)` + `summarize(xs)` stats helpers
  - `TimedYoloProxy`, `TimedClassifier`, `wrap_for_timing(model, bucket)` — non-invasive timing hooks
  - `_build_record(...)` — per-sequence record assembler
  - `run_benchmark_on_model(model, sequence_dirs, warmup) -> dict` — the pure benchmark loop (unit-testable without a CLI)
  - `print_summary(result)` — 4-row terminal table + tail line
  - `_parse_args()` and `main()` — CLI driver
- **Create** `tests/test_benchmark_cpu_latency.py` — two tests as specified in the spec:
  - `test_summarize_percentile_values` — synthetic `[10, 20, 30, 40, 50]` → exact expected values
  - `test_run_benchmark_on_model_happy_path` — in-process fake model + 5 tmp_path sequences, asserts record shape, warmup accounting, timing non-negativity, `total >= yolo + classifier`.

No shared fixture module is created: the test file duplicates the `_fake_yolo_factory` and the test config locally to keep it self-contained (same pattern as the existing `tests/test_model_edge_cases.py`).

---

## Task 1: Stats aggregator (TDD)

**Files:**
- Create: `scripts/benchmark_cpu_latency.py`
- Create: `tests/test_benchmark_cpu_latency.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_benchmark_cpu_latency.py`:

```python
"""Tests for scripts/benchmark_cpu_latency.py."""

from __future__ import annotations

import sys
from pathlib import Path

# The script lives under scripts/ (not a package); import by path.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import benchmark_cpu_latency as bench  # noqa: E402


def test_summarize_percentile_values() -> None:
    stats = bench.summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats == {
        "p50": 30.0,
        "p95": 48.0,
        "mean": 30.0,
        "min": 10.0,
        "max": 50.0,
    }


def test_summarize_raises_on_empty_input() -> None:
    import pytest

    with pytest.raises(ValueError, match="empty"):
        bench.summarize([])
```

- [ ] **Step 1.2: Run the test to verify it fails**

Run: `uv run pytest tests/test_benchmark_cpu_latency.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark_cpu_latency'`.

- [ ] **Step 1.3: Create the script with just the stats helpers**

Create `scripts/benchmark_cpu_latency.py`:

```python
"""CPU latency benchmark for packaged bbox-tube-temporal models.

See ``docs/specs/2026-04-17-cpu-latency-benchmark-design.md``.
"""

from __future__ import annotations


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile. Matches numpy's default ``method='linear'``.

    ``p`` is in [0, 100]. Raises ``ValueError`` on empty input.
    """
    if not xs:
        raise ValueError("percentile() got empty input")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be in [0, 100], got {p!r}")
    sorted_xs = sorted(xs)
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    rank = (len(sorted_xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = rank - lo
    return float(sorted_xs[lo] + frac * (sorted_xs[hi] - sorted_xs[lo]))


def summarize(xs: list[float]) -> dict[str, float]:
    """Return p50/p95/mean/min/max of ``xs``.

    Raises ``ValueError`` on empty input.
    """
    if not xs:
        raise ValueError("summarize() got empty input")
    return {
        "p50": percentile(xs, 50.0),
        "p95": percentile(xs, 95.0),
        "mean": float(sum(xs) / len(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }
```

- [ ] **Step 1.4: Run the test to verify it passes**

Run: `uv run pytest tests/test_benchmark_cpu_latency.py -v`
Expected: PASS (2 passed).

- [ ] **Step 1.5: Lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 1.6: Commit**

```bash
git add scripts/benchmark_cpu_latency.py tests/test_benchmark_cpu_latency.py
git commit -m "feat(bbox-tube-temporal): add stats aggregator for CPU latency benchmark"
```

---

## Task 2: Timing proxies (TDD)

**Files:**
- Modify: `scripts/benchmark_cpu_latency.py` — append timing-proxy classes and `wrap_for_timing`
- Modify: `tests/test_benchmark_cpu_latency.py` — append three proxy tests

- [ ] **Step 2.1: Write the failing proxy tests**

Append to `tests/test_benchmark_cpu_latency.py`:

```python
from unittest.mock import MagicMock

import torch


def test_timed_yolo_proxy_accumulates_bucket() -> None:
    bucket = {"yolo_s": 0.0, "classifier_s": 0.0}
    inner = MagicMock()
    inner.predict.return_value = []
    proxy = bench.TimedYoloProxy(inner, bucket)

    proxy.predict(["/fake/path.jpg"])

    assert inner.predict.call_count == 1
    assert bucket["yolo_s"] > 0.0
    assert bucket["classifier_s"] == 0.0


def test_timed_classifier_accumulates_bucket() -> None:
    bucket = {"yolo_s": 0.0, "classifier_s": 0.0}
    inner = torch.nn.Linear(4, 1)
    proxy = bench.TimedClassifier(inner, bucket)

    proxy(torch.zeros(1, 4))

    assert bucket["classifier_s"] > 0.0
    assert bucket["yolo_s"] == 0.0


def test_wrap_for_timing_installs_both_proxies() -> None:
    model = MagicMock()
    model._yolo = MagicMock()
    model._classifier = torch.nn.Linear(4, 1)
    bucket = {"yolo_s": 0.0, "classifier_s": 0.0}

    bench.wrap_for_timing(model, bucket)

    assert isinstance(model._yolo, bench.TimedYoloProxy)
    assert isinstance(model._classifier, bench.TimedClassifier)
```

- [ ] **Step 2.2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_cpu_latency.py -v`
Expected: the three new tests FAIL with `AttributeError: module 'benchmark_cpu_latency' has no attribute 'TimedYoloProxy'`. The two existing tests still pass.

- [ ] **Step 2.3: Implement the proxies**

Append to `scripts/benchmark_cpu_latency.py`:

```python
import time

import torch


class TimedYoloProxy:
    """Forward ``.predict`` to a wrapped YOLO, accumulating wall-clock into a bucket.

    ``bucket`` is a mutable dict shared across sequences; the driver zeros
    ``bucket["yolo_s"]`` between sequences.
    """

    def __init__(self, wrapped: object, bucket: dict[str, float]) -> None:
        self._wrapped = wrapped
        self._bucket = bucket

    def predict(self, *args: object, **kwargs: object) -> object:
        t0 = time.perf_counter()
        result = self._wrapped.predict(*args, **kwargs)
        self._bucket["yolo_s"] += time.perf_counter() - t0
        return result


class TimedClassifier(torch.nn.Module):
    """Forward-wrap the packaged classifier, accumulating wall-clock into a bucket.

    Subclasses ``nn.Module`` so ``.eval()``, ``.to()``, and parameter iteration
    keep working through the proxy. Captures every forward pass — including the
    prefix-scoring calls inside ``find_first_crossing_trigger`` — because the
    wrapper is installed on ``model._classifier`` itself.
    """

    def __init__(self, wrapped: torch.nn.Module, bucket: dict[str, float]) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.bucket = bucket

    def forward(self, *args: object, **kwargs: object) -> object:
        t0 = time.perf_counter()
        result = self.wrapped(*args, **kwargs)
        self.bucket["classifier_s"] += time.perf_counter() - t0
        return result


def wrap_for_timing(model: object, bucket: dict[str, float]) -> None:
    """Install TimedYoloProxy and TimedClassifier on ``model`` in-place.

    Expected to be called exactly once per benchmark, immediately after
    ``BboxTubeTemporalModel.from_archive``.
    """
    model._yolo = TimedYoloProxy(model._yolo, bucket)
    model._classifier = TimedClassifier(model._classifier, bucket).eval()
```

- [ ] **Step 2.4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_cpu_latency.py -v`
Expected: PASS (5 passed total).

- [ ] **Step 2.5: Lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 2.6: Commit**

```bash
git add scripts/benchmark_cpu_latency.py tests/test_benchmark_cpu_latency.py
git commit -m "feat(bbox-tube-temporal): add timing proxies for YOLO and classifier"
```

---

## Task 3: Benchmark loop (TDD, integration)

**Files:**
- Modify: `scripts/benchmark_cpu_latency.py` — add `_build_record` and `run_benchmark_on_model`
- Modify: `tests/test_benchmark_cpu_latency.py` — add the integration test

- [ ] **Step 3.1: Write the failing integration test**

Append to `tests/test_benchmark_cpu_latency.py`:

```python
from pathlib import Path as _Path

import numpy as np
from PIL import Image as _Image

from bbox_tube_temporal.model import BboxTubeTemporalModel
from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier


BENCHMARK_TEST_CONFIG: dict = {
    "infer": {
        "confidence_threshold": 0.01,
        "iou_nms": 0.2,
        "image_size": 1024,
        "pad_to_min_frames": 0,
        "pad_strategy": "symmetric",
    },
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 4,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 8,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "resnet18",
        "arch": "gru",
        "hidden_dim": 32,
        "num_layers": 1,
        "bidirectional": False,
        "max_frames": 6,
        "pretrained": False,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


def _fake_yolo_factory(per_frame_xywhn):
    """Return a mock YOLO whose ``.predict`` yields fixed detections per frame.

    Mirrors the helper in ``tests/test_model_edge_cases.py``.
    """

    def fake_predict(paths, **_kwargs):
        assert len(paths) == len(per_frame_xywhn)
        results = []
        for boxes in per_frame_xywhn:
            r = MagicMock()
            if not boxes:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self: 0
                r.boxes.xywhn = torch.zeros(0, 4)
                r.boxes.conf = torch.zeros(0)
                r.boxes.cls = torch.zeros(0)
            else:
                n = len(boxes)
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self, n=n: n
                r.boxes.xywhn = torch.tensor(
                    [[c, cy, w, h] for (c, cy, w, h, _) in boxes]
                )
                r.boxes.conf = torch.tensor(
                    [conf for (_, _, _, _, conf) in boxes]
                )
                r.boxes.cls = torch.zeros(n)
            results.append(r)
        return results

    m = MagicMock()
    m.predict.side_effect = fake_predict
    return m


def _make_seq(seq_dir: _Path, n_frames: int) -> _Path:
    img_dir = seq_dir / "images"
    img_dir.mkdir(parents=True)
    for j in range(n_frames):
        arr = np.full((64, 64, 3), fill_value=[180, 30, 30], dtype=np.uint8)
        _Image.fromarray(arr).save(
            img_dir / f"cam_2026-04-17T10-00-{j:02d}.jpg", format="JPEG"
        )
    return seq_dir


def test_run_benchmark_on_model_happy_path(tmp_path: _Path) -> None:
    # 5 sequences × 6 frames, split 3 smoke + 2 fp (the benchmark
    # is label-agnostic; mixing just proves it doesn't care).
    seq_dirs = []
    for i in range(5):
        label = "wildfire" if i < 3 else "fp"
        seq_dir = tmp_path / "sequences" / label / f"seq_{i}"
        _make_seq(seq_dir, n_frames=6)
        seq_dirs.append(seq_dir)

    # Fake YOLO emits one detection per frame so tubes are built and the
    # classifier is exercised (exactly the scenario the spec's integration
    # test pins down).
    per_frame = [(0.5, 0.5, 0.1, 0.1, 0.9)]
    yolo = _fake_yolo_factory([[per_frame[0]] for _ in range(6)])
    classifier = TemporalSmokeClassifier(
        backbone="resnet18", arch="gru", hidden_dim=32, pretrained=False
    )
    model = BboxTubeTemporalModel(
        yolo_model=yolo,
        classifier=classifier,
        config=BENCHMARK_TEST_CONFIG,
        device="cpu",
    )

    result = bench.run_benchmark_on_model(model, seq_dirs, warmup=2)

    assert len(result["records"]) == 5
    assert [r["is_warmup"] for r in result["records"]] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert result["summary"]["num_sequences"] == 3
    assert result["summary"]["num_warmup_skipped"] == 2

    for r in result["records"]:
        assert r["total_s"] >= 0.0
        assert r["yolo_s"] >= 0.0
        assert r["classifier_s"] >= 0.0
        # Timings are subsets of total — the sign is what we assert.
        assert r["total_s"] + 1e-6 >= r["yolo_s"] + r["classifier_s"]

    for r in result["records"][2:]:  # non-warmup
        assert r["yolo_s"] > 0.0, "YOLO proxy must fire on every sequence"
        assert r["classifier_s"] > 0.0, (
            "classifier proxy must fire on every non-warmup sequence (fake "
            "YOLO emits detections so tubes are built)"
        )

    # Summary sub-dicts exist and have the expected shape.
    for key in ("total_ms", "yolo_ms", "classifier_ms", "other_ms", "per_frame_total_ms"):
        assert set(result["summary"][key].keys()) == {
            "p50",
            "p95",
            "mean",
            "min",
            "max",
        }
```

- [ ] **Step 3.2: Run the integration test to verify it fails**

Run: `uv run pytest tests/test_benchmark_cpu_latency.py::test_run_benchmark_on_model_happy_path -v`
Expected: FAIL — `AttributeError: module 'benchmark_cpu_latency' has no attribute 'run_benchmark_on_model'`.

- [ ] **Step 3.3: Implement the benchmark loop**

Append to `scripts/benchmark_cpu_latency.py`:

```python
from pathlib import Path

from bbox_tube_temporal.data import get_sorted_frames


def _build_record(
    *,
    seq_dir: Path,
    num_frames: int,
    num_tubes_kept: int,
    yolo_s: float,
    classifier_s: float,
    total_s: float,
    is_warmup: bool,
) -> dict:
    return {
        "sequence_id": seq_dir.name,
        "num_frames": num_frames,
        "num_tubes_kept": num_tubes_kept,
        "yolo_s": yolo_s,
        "classifier_s": classifier_s,
        "total_s": total_s,
        "is_warmup": is_warmup,
    }


def run_benchmark_on_model(
    model: object,
    sequence_dirs: list[Path],
    *,
    warmup: int,
) -> dict:
    """Run ``model.predict`` on each sequence, accumulating per-sequence timings.

    Installs :class:`TimedYoloProxy` and :class:`TimedClassifier` on ``model``
    in-place, then iterates ``sequence_dirs`` in the given order. The first
    ``warmup`` records are retained but flagged ``is_warmup=True`` and excluded
    from summary aggregates.

    Returns a dict ``{"summary": {...}, "records": [...]}`` matching the
    spec's output schema (minus the top-level ``model_zip`` / ``device``
    fields, which the CLI wrapper fills in).
    """
    bucket = {"yolo_s": 0.0, "classifier_s": 0.0}
    wrap_for_timing(model, bucket)

    records: list[dict] = []
    for i, seq_dir in enumerate(sequence_dirs):
        frame_paths = get_sorted_frames(seq_dir)
        if not frame_paths:
            continue
        frames = model.load_sequence(frame_paths)

        # Zero the bucket right before each prediction so per-sequence
        # timings are independent of prior calls.
        bucket["yolo_s"] = 0.0
        bucket["classifier_s"] = 0.0

        t0 = time.perf_counter()
        output = model.predict(frames)
        total_s = time.perf_counter() - t0

        records.append(
            _build_record(
                seq_dir=seq_dir,
                num_frames=len(frames),
                num_tubes_kept=len(output.details.get("tubes", {}).get("kept", [])),
                yolo_s=bucket["yolo_s"],
                classifier_s=bucket["classifier_s"],
                total_s=total_s,
                is_warmup=i < warmup,
            )
        )

    body = [r for r in records if not r["is_warmup"]]
    if not body:
        raise ValueError(
            f"benchmark produced no non-warmup records "
            f"(got {len(records)} records with warmup={warmup})"
        )

    total_ms = [r["total_s"] * 1000.0 for r in body]
    yolo_ms = [r["yolo_s"] * 1000.0 for r in body]
    classifier_ms = [r["classifier_s"] * 1000.0 for r in body]
    other_ms = [
        (r["total_s"] - r["yolo_s"] - r["classifier_s"]) * 1000.0 for r in body
    ]
    per_frame_total_ms = [
        (r["total_s"] * 1000.0) / r["num_frames"]
        for r in body
        if r["num_frames"] > 0
    ]

    summary = {
        "num_sequences": len(body),
        "num_warmup_skipped": sum(1 for r in records if r["is_warmup"]),
        "total_ms": summarize(total_ms),
        "yolo_ms": summarize(yolo_ms),
        "classifier_ms": summarize(classifier_ms),
        "other_ms": summarize(other_ms),
        "per_frame_total_ms": summarize(per_frame_total_ms),
    }
    return {"summary": summary, "records": records}
```

- [ ] **Step 3.4: Run the integration test to verify it passes**

Run: `uv run pytest tests/test_benchmark_cpu_latency.py::test_run_benchmark_on_model_happy_path -v`
Expected: PASS.

- [ ] **Step 3.5: Run the full test file**

Run: `uv run pytest tests/test_benchmark_cpu_latency.py -v`
Expected: PASS (6 passed).

- [ ] **Step 3.6: Lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 3.7: Commit**

```bash
git add scripts/benchmark_cpu_latency.py tests/test_benchmark_cpu_latency.py
git commit -m "feat(bbox-tube-temporal): add run_benchmark_on_model loop"
```

---

## Task 4: CLI + JSON output + terminal table

**Files:**
- Modify: `scripts/benchmark_cpu_latency.py` — append `_parse_args`, `print_summary`, `main`

No new tests: Task 3's `run_benchmark_on_model` is the logic under test. The CLI is a thin wrapper whose correctness is visible in Task 5's manual smoke run.

- [ ] **Step 4.1: Implement the CLI**

Append to `scripts/benchmark_cpu_latency.py`:

```python
import argparse
import json
import statistics

from tqdm import tqdm

from bbox_tube_temporal.data import list_sequences
from bbox_tube_temporal.model import BboxTubeTemporalModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zip", type=Path, required=True)
    parser.add_argument("--sequences-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Torch device. Defaults to cpu for benchmark reproducibility.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap on number of sequences (first N in sorted order). None = all.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of leading sequences whose timings are retained but "
        "excluded from summary aggregates.",
    )
    return parser.parse_args()


def print_summary(result: dict) -> None:
    """4-row table (total / yolo / classifier / other) + tail line."""
    summary = result["summary"]
    records = result["records"]
    body = [r for r in records if not r["is_warmup"]]

    rows = [
        ("total", summary["total_ms"]),
        ("yolo", summary["yolo_ms"]),
        ("classifier", summary["classifier_ms"]),
        ("other", summary["other_ms"]),
    ]
    print(f"{'metric':<12} {'p50 (ms)':>12} {'p95 (ms)':>12} {'mean (ms)':>12}")
    print("-" * 52)
    for name, stats in rows:
        print(
            f"{name:<12} {stats['p50']:>12.2f} {stats['p95']:>12.2f} "
            f"{stats['mean']:>12.2f}"
        )
    median_frames = statistics.median(r["num_frames"] for r in body)
    median_tubes = statistics.median(r["num_tubes_kept"] for r in body)
    median_total_s = summary["total_ms"]["p50"] / 1000.0
    print(
        f"\n{summary['num_sequences']} sequences "
        f"(excluded {summary['num_warmup_skipped']} warmup), "
        f"median {median_total_s:.2f} s/seq, "
        f"{median_frames:g} frames median, "
        f"{median_tubes:g} tubes kept median"
    )


def main() -> None:
    args = _parse_args()

    model = BboxTubeTemporalModel.from_archive(args.model_zip, device=args.device)

    sequences = list_sequences(args.sequences_dir)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]

    # Wrap tqdm around the generator feeding run_benchmark_on_model by
    # passing a tqdm-decorated list; the loop inside the function runs
    # linearly and does not need a progress bar of its own.
    result = run_benchmark_on_model(
        model,
        list(tqdm(sequences, desc=f"bench {args.model_zip.name}", unit="seq")),
        warmup=args.warmup,
    )
    # Add top-level context to the summary so a standalone JSON is self-describing.
    result["summary"]["model_zip"] = str(args.model_zip)
    result["summary"]["device"] = args.device

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print_summary(result)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Verify the script imports cleanly**

Run: `uv run python -c "import sys; sys.path.insert(0, 'scripts'); import benchmark_cpu_latency; print(benchmark_cpu_latency.__doc__)"`
Expected: prints the module docstring (the first line of `__doc__`). No import errors.

- [ ] **Step 4.3: Verify `--help` renders**

Run: `uv run python scripts/benchmark_cpu_latency.py --help`
Expected: argparse help text listing all six flags (`--model-zip`, `--sequences-dir`, `--output`, `--device`, `--max-sequences`, `--warmup`).

- [ ] **Step 4.4: Run the full test file again**

Run: `uv run pytest tests/test_benchmark_cpu_latency.py -v`
Expected: PASS (6 passed). The new CLI code does not affect existing tests.

- [ ] **Step 4.5: Lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 4.6: Commit**

```bash
git add scripts/benchmark_cpu_latency.py
git commit -m "feat(bbox-tube-temporal): add CLI + JSON output + terminal table"
```

---

## Task 5: Manual smoke run

**Files:** none — verification only.

The goal of this task is to run the script end-to-end on a small slice of real data, confirm the JSON has the expected shape, and eyeball the terminal summary. No commit unless the script needs a bug fix.

- [ ] **Step 5.1: Run the script on gru_convnext_finetune / val / CPU / 10 sequences, warmup=3**

Run:

```bash
uv run python scripts/benchmark_cpu_latency.py \
  --model-zip data/06_models/gru_convnext_finetune/model.zip \
  --sequences-dir data/01_raw/datasets/val \
  --output /tmp/bench-gru-cpu.json \
  --device cpu \
  --max-sequences 10 \
  --warmup 3
```

Expected: tqdm bar reaches 10/10; terminal table prints four rows (total / yolo / classifier / other) with ms values; tail line reports `7 sequences (excluded 3 warmup), median N.NN s/seq, ...`. `/tmp/bench-gru-cpu.json` exists.

- [ ] **Step 5.2: Validate the JSON shape**

Run:

```bash
uv run python -c "
import json
d = json.loads(open('/tmp/bench-gru-cpu.json').read())
s = d['summary']
assert s['num_sequences'] == 7, s
assert s['num_warmup_skipped'] == 3, s
assert s['device'] == 'cpu'
assert s['model_zip'].endswith('gru_convnext_finetune/model.zip')
for k in ('total_ms', 'yolo_ms', 'classifier_ms', 'other_ms', 'per_frame_total_ms'):
    assert set(s[k]) == {'p50', 'p95', 'mean', 'min', 'max'}, (k, s[k])
assert len(d['records']) == 10
assert sum(1 for r in d['records'] if r['is_warmup']) == 3
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 5.3: Run the script on vit_dinov2_finetune / val / CPU / 10 sequences**

The headline experiment the spec was written for.

Run:

```bash
uv run python scripts/benchmark_cpu_latency.py \
  --model-zip data/06_models/vit_dinov2_finetune/model.zip \
  --sequences-dir data/01_raw/datasets/val \
  --output /tmp/bench-vit-cpu.json \
  --device cpu \
  --max-sequences 10 \
  --warmup 3
```

Expected: same shape as step 5.1; median seconds/sequence likely larger than GRU+ConvNeXt (this is the question the spec was written to answer).

- [ ] **Step 5.4 (optional): Full val on CPU for the ViT variant**

Skip unless the 10-sequence run looks plausible. Reruns without `--max-sequences`:

```bash
uv run python scripts/benchmark_cpu_latency.py \
  --model-zip data/06_models/vit_dinov2_finetune/model.zip \
  --sequences-dir data/01_raw/datasets/val \
  --output data/08_reporting/benchmarks/vit_dinov2_finetune-val-cpu.json \
  --device cpu \
  --warmup 3
```

Expected: runs unattended; produces the full-val JSON under `08_reporting/benchmarks/`. Time estimate: 15–45 min depending on machine.

- [ ] **Step 5.5 (optional): Corresponding GRU+ConvNeXt full val on CPU for side-by-side comparison**

```bash
uv run python scripts/benchmark_cpu_latency.py \
  --model-zip data/06_models/gru_convnext_finetune/model.zip \
  --sequences-dir data/01_raw/datasets/val \
  --output data/08_reporting/benchmarks/gru_convnext_finetune-val-cpu.json \
  --device cpu \
  --warmup 3
```

Expected: full-val JSON; compare against the ViT output to answer "is ViT materially slower on CPU".

---

## Self-review checklist

**1. Spec coverage.** Each spec requirement has a task:

| Spec section                            | Task(s)                                |
| --------------------------------------- | -------------------------------------- |
| Script + interface (`--model-zip`, ...) | Task 4 (`_parse_args`, `main`)         |
| Timing methodology (proxies)            | Task 2 (`TimedYoloProxy`, `TimedClassifier`) |
| Bucket discipline / end-to-end timing   | Task 3 (`run_benchmark_on_model`)      |
| Warmup                                  | Task 3 (`is_warmup` flag + exclusion)  |
| Output JSON schema                      | Task 3 (record shape) + Task 4 (model_zip/device) |
| Terminal summary table                  | Task 4 (`print_summary`)               |
| Stats aggregation (linear interpolation) | Task 1 (`percentile`, `summarize`)    |
| Stats aggregator unit test              | Task 1                                 |
| Integration test                        | Task 3                                 |
| Usage example (vit_dinov2, 10 seqs)     | Task 5                                 |

**2. Placeholder scan.** No "TBD", "TODO", or "similar to Task N" — every code block is complete. No `handle edge cases` / `add validation` — explicit `ValueError` on empty input is the only edge we guard, documented in Task 1 and asserted in the test.

**3. Type consistency.** The shared bucket dict uses keys `yolo_s` and `classifier_s` everywhere (Tasks 2, 3). Record keys are `yolo_s`, `classifier_s`, `total_s`, `num_frames`, `num_tubes_kept`, `sequence_id`, `is_warmup` consistently across the code and the integration test. Summary keys `total_ms`, `yolo_ms`, `classifier_ms`, `other_ms`, `per_frame_total_ms` match the spec's JSON schema and the CLI's self-description. `wrap_for_timing` signature `(model, bucket)` consistent between Task 2's definition and Task 3's call site.

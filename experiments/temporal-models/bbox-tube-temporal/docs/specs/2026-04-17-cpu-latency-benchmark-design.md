# CPU latency benchmark for packaged bbox-tube-temporal models

**Status:** Proposed
**Date:** 2026-04-17
**Scope:** `experiments/temporal-models/bbox-tube-temporal/`

## Context

The `vit_dinov2_finetune` variant is the current best packaged bbox-tube-
temporal model. Pyronear's deployment story assumes a server GPU, but we
do not know whether this variant is viable on CPU-only hardware, nor at
what latency. This document specifies a standalone benchmark script that
answers two questions for any packaged model:

1. **Does it run end-to-end on CPU?**
2. **What is the per-sequence latency, and is it dominated by YOLO or by
   the temporal classifier?**

The split matters because YOLO and the temporal classifier use very
different architectures, and CPU optimisation targets (threading,
quantisation, ONNX export) differ between them.

## Goals

- A single reusable script that benchmarks any packaged `model.zip` on a
  chosen device, with the val (or train) sequence directory as input.
- Per-sequence wall-clock total, plus a YOLO / classifier split.
- Aggregate statistics (p50, p95, mean, min, max) and per-frame
  normalisation so results compare cleanly across sequences of different
  length.
- Warmup handling so the first few (slow) iterations do not poison
  summary stats.
- JSON output for later inspection and a short terminal summary for
  immediate feedback.

## Non-goals

- A DVC-tracked stage. The script is a diagnostic tool; we do not want
  benchmark numbers invalidated on every code change.
- Cross-variant comparison inside the script. Callers run the script
  twice and compare two JSON outputs by hand.
- Modifying `BboxTubeTemporalModel.predict()` or
  `src/bbox_tube_temporal/inference.py` to add timing hooks. Production
  code stays untouched.
- Benchmarking intermediate stages beyond YOLO vs classifier (e.g.,
  separating tube building from patch cropping from post-processing).
  That staged breakdown is a future extension if the YOLO-vs-classifier
  split proves insufficient.
- Controlling for OS noise (cgroup pinning, performance governors, etc.).
  The user is expected to run the benchmark on a quiet machine.

## Design

### Script

`scripts/benchmark_cpu_latency.py`.

```
--model-zip <path>            required     packaged model.zip
--sequences-dir <path>        required     e.g. data/01_raw/datasets/val
--output <path>               required     JSON output path
--device {cpu,cuda,mps}       default cpu
--max-sequences N             default: all
--warmup K                    default 3
```

Iteration order is the sorted output of `list_sequences(sequences_dir)`
(deterministic). `--max-sequences N` takes the first N.

### Timing methodology

The benchmark wraps the two learned components with timing proxies
immediately after loading the packaged model, before any `predict()`
call. This keeps `model.py` and `inference.py` untouched.

```python
class TimedYoloProxy:
    """Forwards .predict to the wrapped YOLO, accumulating wall-clock."""
    def __init__(self, wrapped, bucket):
        self._wrapped = wrapped
        self._bucket = bucket

    def predict(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = self._wrapped.predict(*args, **kwargs)
        self._bucket["yolo_s"] += time.perf_counter() - t0
        return result


class TimedClassifier(torch.nn.Module):
    """Forwards to the wrapped classifier, accumulating wall-clock.

    Subclasses nn.Module so ``.eval()`` / ``.to()`` / parameter iteration
    continue to work if any other code path expects it.
    """
    def __init__(self, wrapped, bucket):
        super().__init__()
        self.wrapped = wrapped
        self.bucket = bucket

    def forward(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = self.wrapped(*args, **kwargs)
        self.bucket["classifier_s"] += time.perf_counter() - t0
        return result


def wrap_for_timing(model, bucket):
    model._yolo = TimedYoloProxy(model._yolo, bucket)
    model._classifier = TimedClassifier(model._classifier, bucket).eval()
```

**Why this works.** `model._yolo` is used only via its `.predict()`
method (`run_yolo_on_frames` calls `yolo_model.predict(paths, ...)`).
`model._classifier` is used as a callable both in `score_tubes` and
inside `find_first_crossing_trigger` — so wrapping the classifier
captures *every* forward pass, including the prefix-scoring calls that a
monkey-patch on `score_tubes` alone would miss.

**Bucket discipline.** The benchmark allocates a single mutable dict
`bucket = {"yolo_s": 0.0, "classifier_s": 0.0}` shared with both
wrappers. Before each sequence, the benchmark zeros the bucket. After
each sequence, it copies the bucket values into that sequence's record.

**End-to-end timing.** The benchmark also measures
`time.perf_counter()` around the outer `model.predict(frames)` call, to
capture total wall-clock including tube building, patch cropping, PIL
decode, and post-processing. `other_s = total_s - yolo_s -
classifier_s` is computed in post-processing.

### Warmup

The first `--warmup K` sequences are executed identically to the rest,
but their records carry `"is_warmup": true` and are excluded from
summary aggregates. Keeping them in `records[]` makes it possible to
diagnose abnormal warmup behaviour after the fact (e.g., whether warmup
was "hot enough").

### Output

JSON file at `--output`:

```json
{
  "summary": {
    "model_zip": "...",
    "device": "cpu",
    "num_sequences": 315,
    "num_warmup_skipped": 3,
    "total_ms":          {"p50": ..., "p95": ..., "mean": ..., "min": ..., "max": ...},
    "yolo_ms":           {"p50": ..., "p95": ..., "mean": ..., "min": ..., "max": ...},
    "classifier_ms":     {"p50": ..., "p95": ..., "mean": ..., "min": ..., "max": ...},
    "other_ms":          {"p50": ..., "p95": ..., "mean": ..., "min": ..., "max": ...},
    "per_frame_total_ms":{"p50": ..., "p95": ..., "mean": ..., "min": ..., "max": ...}
  },
  "records": [
    {
      "sequence_id": "…",
      "num_frames": 12,
      "num_tubes_kept": 2,
      "yolo_s": 0.34,
      "classifier_s": 1.21,
      "total_s": 1.68,
      "is_warmup": false
    },
    ...
  ]
}
```

Terminal summary prints a 4-row table (total / yolo / classifier / other)
with p50, p95, mean columns, plus one tail line:

```
315 sequences, median 1.68 s/seq, 12 frames median, 2 tubes kept median
```

### Stats aggregation

`percentile(xs, p)` uses linear interpolation between the two nearest
ranks (matches `numpy.percentile(xs, p, method="linear")`). An empty
input (fewer than `warmup + 1` sequences) raises `ValueError` — the
benchmark cannot summarise nothing.

### Testing

Two tests in `tests/test_benchmark_cpu_latency.py`:

1. **Stats aggregator unit test.** Synthetic list
   `[10, 20, 30, 40, 50]` (ms). Assert
   `summarize([10, 20, 30, 40, 50]) == {"p50": 30.0, "p95": 48.0,
   "mean": 30.0, "min": 10.0, "max": 50.0}`.
2. **Integration test.** Stub YOLO (`MagicMock` with a `predict`
   returning **one non-empty detection per frame** so tubes are built
   and the classifier is exercised), tiny classifier
   (`TemporalSmokeClassifier(...)` with `pretrained=False`). Reuse the
   `red_frames` on-disk fixture pattern from
   `tests/test_model_edge_cases.py` (writes tiny JPEGs to tmp_path).
   Feed 5 fake sequences (6 frames each). Assert:
   - `records[]` has 5 entries.
   - The first `warmup=2` records have `is_warmup=True`.
   - The summary counts 3 aggregated sequences.
   - `yolo_s > 0` and `classifier_s > 0` for every non-warmup record
     (proves both wrappers fire and both buckets accumulate).
   - `total_s >= yolo_s + classifier_s` for every record (timings are
     subsets of total, not a partition beyond that — only the sign is
     asserted).

## Usage example

```bash
uv run python scripts/benchmark_cpu_latency.py \
  --model-zip data/06_models/vit_dinov2_finetune/model.zip \
  --sequences-dir data/01_raw/datasets/val \
  --output data/08_reporting/benchmarks/vit_dinov2_finetune-val-cpu.json \
  --device cpu
```

For a quick iteration pass during script development:

```bash
uv run python scripts/benchmark_cpu_latency.py \
  ... --max-sequences 30 --warmup 3
```

Compare variants by running twice with different `--model-zip` and
diffing the two JSON outputs.

## Future extensions (explicit non-goals today)

- **Staged breakdown.** Add tube-building and patch-cropping timers if
  the "other" bucket proves surprisingly large.
- **Multi-threading sweep.** `torch.set_num_threads(...)` over
  `{1, 2, 4, 8}` to chart latency vs. CPU cores.
- **ONNX / quantisation.** Export the classifier to ONNX and re-run the
  benchmark to quantify inference-time gains.
- **Cross-device table.** Run the benchmark on `cpu`, `cuda`, `mps` and
  produce a combined table — currently left to the caller.

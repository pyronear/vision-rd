"""CPU (and any-device) latency benchmark for the packaged temporal model.

Library module: stats helpers, timing proxies, and the ``run_benchmark_on_model``
driver live here. ``scripts/benchmark_cpu_latency.py`` is the thin CLI wrapper.

See ``docs/specs/2026-04-17-cpu-latency-benchmark-design.md``.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from .data import get_sorted_frames


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile.

    Matches ``numpy.percentile(..., method='linear')``. ``p`` is in [0, 100].
    Raises ``ValueError`` on empty input.
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
    """Install :class:`TimedYoloProxy` and :class:`TimedClassifier` on ``model``.

    Expected to be called exactly once per benchmark, immediately after
    ``BboxTubeTemporalModel.from_archive``.
    """
    model._yolo = TimedYoloProxy(model._yolo, bucket)
    model._classifier = TimedClassifier(model._classifier, bucket).eval()


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

    Returns a dict ``{"summary": {...}, "records": [...]}`` matching the spec's
    output schema (minus the top-level ``model_zip`` / ``device`` fields, which
    the CLI wrapper fills in).
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

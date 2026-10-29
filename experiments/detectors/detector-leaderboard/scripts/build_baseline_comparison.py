"""Generate baseline_comparison.md — every model vs the production baseline.

Reads the aggregated ``data/08_reporting/leaderboard.json`` and writes a markdown
report comparing each model against the baseline (``yolo11s-nimble-narwhal-v6.0.0``)
on three metrics, one table each:

1. **F1** (test box-F1, higher is better)
2. **False-positive rate** (image-level FPR on background frames, lower is better)
3. **Time to detection** (per-frame inference latency in ms, lower is better)

"Time to detection" here is the model's inference latency from the profiling
step — the time the model takes to produce detections on a frame. (The
wildfire sense of time-to-detection — minutes from ignition to first alarm —
is not computed anywhere in this pipeline; it would need event-grouped temporal
sequences with ignition timestamps.)

    uv run python scripts/build_baseline_comparison.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "data" / "08_reporting" / "leaderboard.json"
OUT = ROOT / "baseline_comparison.md"
BASELINE = "yolo11s-nimble-narwhal-v6.0.0"


def latency(model: dict) -> float | None:
    return (model.get("profile") or {}).get("latency_ms")


def fmt_delta(x: float, decimals: int) -> str:
    """Signed delta, e.g. +0.0111 / -0.0598 / 0 for the baseline itself."""
    if abs(x) < 10 ** (-decimals) / 2:
        return "0"
    return f"{x:+.{decimals}f}"


def fmt_pct(x: float) -> str:
    if abs(x) < 0.05:
        return "0%"
    return f"{x:+.1f}%"


def verdict(is_baseline: bool, better: bool, tie: bool) -> str:
    if is_baseline:
        return "— baseline"
    if tie:
        return "= tie"
    return "▲ better" if better else "▼ worse"


def higher_better_table(models: list[dict], base: dict, key: str, col: str) -> str:
    """Ranked table for a higher-is-better metric (e.g. F1, recall)."""
    rows = sorted(models, key=lambda m: m[key], reverse=True)
    lines = [
        f"| Rank | Model | {col} | Δ {col} | % change | vs baseline |",
        "| ---: | :--- | ---: | ---: | ---: | :--- |",
    ]
    beat = 0
    for i, m in enumerate(rows, 1):
        d = m[key] - base[key]
        is_base = m["model_name"] == BASELINE
        better = d > 0
        tie = abs(d) < 5e-5
        if better and not is_base:
            beat += 1
        name = f"**{m['model_name']}**" if is_base else m["model_name"]
        pct = 100 * d / base[key] if base[key] else 0.0
        lines.append(
            f"| {i} | {name} | {m[key]:.4f} | {fmt_delta(d, 4)} | "
            f"{fmt_pct(pct)} | {verdict(is_base, better, tie)} |"
        )
    summary = (
        f"\n*{beat} of {len(rows) - 1} models beat the baseline's {col} "
        f"({base[key]:.4f}).*\n"
    )
    return "\n".join(lines) + "\n" + summary


def fpr_table(models: list[dict], base: dict) -> str:
    """Lower image-FPR is better."""
    rows = sorted(models, key=lambda m: m["image_fpr"])
    lines = [
        "| Rank | Model | Image FPR | Δ FPR | % change | vs baseline |",
        "| ---: | :--- | ---: | ---: | ---: | :--- |",
    ]
    beat = 0
    for i, m in enumerate(rows, 1):
        d = m["image_fpr"] - base["image_fpr"]
        is_base = m["model_name"] == BASELINE
        better = d < 0  # fewer false alarms
        tie = abs(d) < 5e-5
        if better and not is_base:
            beat += 1
        name = f"**{m['model_name']}**" if is_base else m["model_name"]
        pct = 100 * d / base["image_fpr"] if base["image_fpr"] else 0.0
        lines.append(
            f"| {i} | {name} | {m['image_fpr']:.4f} | {fmt_delta(d, 4)} | "
            f"{fmt_pct(pct)} | {verdict(is_base, better, tie)} |"
        )
    summary = (
        f"\n*{beat} of {len(rows) - 1} models have a lower false-positive rate "
        f"than the baseline ({base['image_fpr']:.4f}).*\n"
    )
    return "\n".join(lines) + "\n" + summary


def latency_table(models: list[dict], base: dict) -> str:
    """Lower inference latency is better; SAHI variants have no standalone profile."""
    with_lat = [m for m in models if latency(m) is not None]
    without = [m["model_name"] for m in models if latency(m) is None]
    rows = sorted(with_lat, key=lambda m: latency(m))
    base_lat = latency(base)
    lines = [
        "| Rank | Model | Latency (ms) | Δ ms | Relative | vs baseline | Input |",
        "| ---: | :--- | ---: | ---: | ---: | :--- | ---: |",
    ]
    beat = 0
    for i, m in enumerate(rows, 1):
        lat = latency(m)
        d = lat - base_lat
        is_base = m["model_name"] == BASELINE
        better = d < 0  # faster
        tie = abs(d) < 5e-3
        if better and not is_base:
            beat += 1
        name = f"**{m['model_name']}**" if is_base else m["model_name"]
        ratio = lat / base_lat if base_lat else 0.0
        rel = "1.00× (baseline)" if is_base else f"{ratio:.2f}× baseline"
        img = (m.get("profile") or {}).get("image_size", "")
        lines.append(
            f"| {i} | {name} | {lat:.2f} | {fmt_delta(d, 2)} | {rel} | "
            f"{verdict(is_base, better, tie)} | {img} |"
        )
    summary = (
        f"\n*{beat} of {len(rows) - 1} profiled models are faster than the "
        f"baseline ({base_lat:.2f} ms @ input "
        f"{(base.get('profile') or {}).get('image_size', '?')}).*\n"
    )
    if without:
        summary += (
            "\n*Excluded (no standalone latency profile — SAHI wraps a base "
            "model with tiled inference, so its cost is the base model's "
            f"latency × the number of tiles): {', '.join(sorted(without))}.*\n"
        )
    return "\n".join(lines) + "\n" + summary


def build() -> None:
    models = json.loads(LEADERBOARD.read_text())
    base = next((m for m in models if m["model_name"] == BASELINE), None)
    if base is None:
        raise SystemExit(f"baseline {BASELINE!r} not found in {LEADERBOARD}")

    base_img = (base.get("profile") or {}).get("image_size", "?")
    md = f"""# Baseline comparison

Each model compared against the production baseline **`{BASELINE}`**
(F1 {base["f1"]:.4f}, recall {base["recall"]:.4f}, image FPR
{base["image_fpr"]:.4f}, latency {latency(base):.2f} ms @ input {base_img}), on
four metrics — one table each. All quality metrics are on the **test** split at
each model's own validation-selected confidence threshold (IoU-match 0.1,
class-agnostic); latency is the profiled per-frame inference time.

Generated by `scripts/build_baseline_comparison.py` from
`data/08_reporting/leaderboard.json`.

## 1. F1 vs baseline (higher is better)

Box-level F1 on the test split. Δ is the model's F1 minus the baseline's.

{higher_better_table(models, base, "f1", "F1")}
## 2. Recall vs baseline (higher is better)

Box-level recall on the test split — the fraction of ground-truth smoke boxes
detected. Δ is the model's recall minus the baseline's.

{higher_better_table(models, base, "recall", "Recall")}
## 3. False-positive rate vs baseline (lower is better)

Image-level FPR — the fraction of background (no-smoke) frames on which the
model raises at least one detection. This is the operational cost metric: false
alarms. A negative Δ / % change means fewer false alarms than the baseline.

{fpr_table(models, base)}
## 4. Time to detection vs baseline (lower is better)

Per-frame inference latency (the model's time to produce detections), from the
profiling step. "Relative" is the model's latency as a multiple of the
baseline's. Note the baseline runs at input 1024 and the DEIMv2 models at 640,
so latency is not input-size-normalized.

{latency_table(models, base)}"""

    OUT.write_text(md)
    print(f"wrote {OUT} ({len(models)} models vs baseline {BASELINE})")


if __name__ == "__main__":
    build()

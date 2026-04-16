"""Offline simulation of a confidence-threshold lift on packaged predictions.

Reads predictions.json (post-B1, with ``kept_tubes`` + ``confidence`` per entry)
and, for a sweep of candidate YOLO confidence thresholds, approximates what
the sequence-level decision would look like if low-confidence detections
were filtered upstream.

**Approximation.** The real `build_tubes` step would re-cluster detections
after filtering — tubes could reshape. This script instead keeps each tube
as-is and drops tubes whose *max entry confidence* falls below τ. It is a
useful upper-bound on "which tubes survive at τ" and a clean direct answer
to "which sequences lose every tube".

For each τ it reports, split by GT class:
  - n_seq_no_tubes_now   — sequences with zero kept tubes (baseline FN risk).
  - n_seq_all_tubes_drop — sequences where every tube would drop at τ
    (guaranteed predicted-negative).
  - simulated PR: treat sequence as positive iff any surviving tube has
    ``logit >= threshold`` (same decision rule used by the model).

Usage:
    uv run python scripts/simulate_confidence_filter.py \\
        --predictions-path data/08_reporting/val/packaged/gru_convnext_finetune/predictions.json \\
        --taus 0.01 0.05 0.10 0.15 0.20 0.25
"""

import argparse
import math
from pathlib import Path

from bbox_tube_temporal.aggregation_analysis import load_predictions


def _tube_max_confidence(tube: dict) -> float:
    """Max confidence across the tube's detected entries; 0.0 if no detections."""
    confs = [
        e["confidence"]
        for e in tube["entries"]
        if e["confidence"] is not None
    ]
    return max(confs) if confs else 0.0


def _simulate_at_tau(
    records: list[dict],
    tau: float,
) -> dict:
    """Simulate the pipeline if YOLO detections below tau were filtered out.

    Uses the per-record baked threshold (it does not vary per tube).
    """
    by_class: dict[str, dict] = {
        "smoke": {
            "n_total": 0,
            "n_seq_no_tubes_now": 0,
            "n_seq_all_tubes_drop": 0,
            "tp": 0, "fn": 0,
            "baseline_tp": 0, "baseline_fn": 0,
        },
        "fp": {
            "n_total": 0,
            "n_seq_no_tubes_now": 0,
            "n_seq_all_tubes_drop": 0,
            "fp": 0, "tn": 0,
            "baseline_fp": 0, "baseline_tn": 0,
        },
    }

    for r in records:
        cls = r["label"]
        bucket = by_class[cls]
        bucket["n_total"] += 1

        # Baseline decision (as persisted).
        if r["is_positive"]:
            if cls == "smoke":
                bucket["baseline_tp"] += 1
            else:
                bucket["baseline_fp"] += 1
        else:
            if cls == "smoke":
                bucket["baseline_fn"] += 1
            else:
                bucket["baseline_tn"] += 1

        kept = r["kept_tubes"]
        if not kept:
            bucket["n_seq_no_tubes_now"] += 1
            # No tubes → sequence stays negative regardless of τ.
            if cls == "smoke":
                bucket["fn"] += 1
            else:
                bucket["tn"] += 1
            continue

        threshold = r["threshold"]
        surviving_logits = [
            t["logit"] for t in kept if _tube_max_confidence(t) >= tau
        ]
        if not surviving_logits:
            bucket["n_seq_all_tubes_drop"] += 1
            sim_positive = False
        else:
            sim_positive = max(surviving_logits) >= threshold

        if sim_positive:
            if cls == "smoke":
                bucket["tp"] += 1
            else:
                bucket["fp"] += 1
        else:
            if cls == "smoke":
                bucket["fn"] += 1
            else:
                bucket["tn"] += 1

    return by_class


def _format_summary(simulated_by_tau: dict[float, dict]) -> str:
    lines: list[str] = []
    lines.append(
        "| tau | class | n_total | no_tubes_now | all_drop | "
        "TP/FP | FN/TN | precision | recall |"
    )
    lines.append(
        "|-----|-------|---------|--------------|----------|-------|-------|-----------|--------|"
    )
    for tau, by_class in simulated_by_tau.items():
        for cls in ("smoke", "fp"):
            b = by_class[cls]
            if cls == "smoke":
                primary = f"{b['tp']}"
                secondary = f"{b['fn']}"
            else:
                primary = f"{b['fp']}"
                secondary = f"{b['tn']}"
            # Aggregate precision/recall across both classes once per tau.
            all_tp = by_class["smoke"]["tp"]
            all_fp = by_class["fp"]["fp"]
            all_fn = by_class["smoke"]["fn"]
            all_tn = by_class["fp"]["tn"]
            p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else float("nan")
            r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else float("nan")
            lines.append(
                f"| {tau:.2f} | {cls:5s} | {b['n_total']:7d} | "
                f"{b['n_seq_no_tubes_now']:12d} | {b['n_seq_all_tubes_drop']:8d} | "
                f"{primary:>5} | {secondary:>5} | "
                f"{p:9.4f} | {r:6.4f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_predictions(args.predictions_path)
    assert records and "kept_tubes" in records[0], (
        "predictions.json is missing 'kept_tubes'; re-run evaluate_packaged "
        "after B1 lands so per-tube details are persisted."
    )

    simulated: dict[float, dict] = {}
    for tau in sorted(args.taus):
        if not (0.0 <= tau <= 1.0) or math.isnan(tau):
            raise ValueError(f"tau must be in [0, 1], got {tau!r}")
        simulated[tau] = _simulate_at_tau(records, tau)

    report = (
        f"# Confidence-threshold filter simulation\n\n"
        f"Input: `{args.predictions_path}`\n\n"
        f"Approximation: drop tubes whose max entry confidence < τ; keep the\n"
        f"rest as-is. Re-apply the baked classifier threshold. See script\n"
        f"docstring for caveats.\n\n"
        + _format_summary(simulated)
        + "\n"
    )

    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()

"""Sweep the sequence-level classifier threshold over packaged predictions.

Reads a predictions.json (with ``tube_logits`` per sequence) and reports
precision / recall / F1 across a range of thresholds using the deployed
``max(tube_logits)`` aggregation rule. Highlights the baked threshold and
filters to a reasonable band (R ≥ min-recall AND P ≥ min-precision) to
keep the output compact.

Usage:
    uv run python scripts/sweep_classifier_threshold.py \\
        --predictions-path data/08_reporting/val/packaged_ablation_c1_conf_0_10/gru_convnext_finetune/predictions.json \\
        --min-recall 0.90 \\
        --min-precision 0.85
"""

import argparse
import math
from pathlib import Path

import numpy as np

from bbox_tube_temporal.aggregation_analysis import (
    build_scores_and_labels,
    load_predictions,
    metrics_at_threshold,
)


def _prob_to_logit(p: float) -> float:
    return math.log(p / (1 - p))


def _logit_to_prob(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--min-precision", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_predictions(args.predictions_path)
    y, s = build_scores_and_labels(records, rule="max", k=1)

    baked_prob = records[0]["threshold"]
    baked_logit = _prob_to_logit(baked_prob)

    pos_scores = sorted(float(v) for v in s[(y == 1) & np.isfinite(s)])
    candidates = sorted(
        set(
            [round(x, 4) for x in pos_scores]
            + [baked_logit - 2.0, baked_logit - 1.0, baked_logit, baked_logit + 1.0]
        )
    )

    header = (
        f"# Classifier-threshold sweep\n\n"
        f"Input: `{args.predictions_path}`\n\n"
        f"Aggregation: `max(tube_logits) >= threshold` (deployed rule).\n\n"
        f"Filter: rows with P ≥ {args.min_precision} AND R ≥ {args.min_recall}.\n\n"
        f"Baked threshold: prob={baked_prob:.4f}  "
        f"logit={baked_logit:+.4f}\n\n"
        "| thr_logit | thr_prob | precision | recall | F1 | TP | FP | FN | marker |\n"
        "|-----------|----------|-----------|--------|----|----|----|----|--------|\n"
    )

    lines = []
    for thr in candidates:
        m = metrics_at_threshold(y, s, threshold=thr)
        if m["precision"] < args.min_precision or m["recall"] < args.min_recall:
            continue
        marker = "**baked**" if abs(thr - baked_logit) < 1e-3 else ""
        lines.append(
            f"| {thr:+.4f} | {_logit_to_prob(thr):.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} | {marker} |"
        )

    report = header + "\n".join(lines) + "\n"
    print(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"\nWrote report to {args.output}")


if __name__ == "__main__":
    main()

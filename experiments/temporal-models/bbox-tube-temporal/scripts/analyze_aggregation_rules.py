"""Offline aggregation-rule analysis on evaluate_packaged predictions.

For each (variant, split), loads predictions.json and explores
alternative sequence-level aggregation rules over per-tube logits.
Writes a markdown report ranked by precision @ target_recall.

Usage:
    uv run python scripts/analyze_aggregation_rules.py \\
        --reporting-dir data/08_reporting \\
        --output data/08_reporting/aggregation_ablation.md \\
        --target-recall 0.95
"""

import argparse
from pathlib import Path

from bbox_tube_temporal.aggregation_analysis import (
    load_predictions,
    summarize_rule,
)

VARIANTS = ("gru_convnext_finetune", "vit_dinov2_finetune")
SPLITS = ("train", "val")
RULE_GRID = (
    ("max", 1),
    ("top_k_mean", 2),
    ("top_k_mean", 3),
)


def _format_row(row: dict) -> str:
    return (
        f"| {row['variant']} | {row['split']} | {row['rule']} | {row['k']} | "
        f"{row['threshold']:.4f} | {row['precision']:.4f} | "
        f"{row['recall']:.4f} | {row['f1']:.4f} | {row['fpr']:.4f} | "
        f"{row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |"
    )


def _render_report(rows: list[dict], target_recall: float) -> str:
    header = (
        "# Aggregation-rule ablation\n\n"
        f"Target recall for threshold search: **{target_recall}**.\n\n"
        "One threshold is chosen per (variant, split, rule) to hit the target recall;\n"
        "precision/FPR/etc. are reported at that threshold.\n\n"
        "| variant | split | rule | k | threshold | precision | recall | F1 | FPR | TP | FP | FN | TN |\n"
        "|---------|-------|------|---|-----------|-----------|--------|----|----|----|----|----|----|\n"
    )
    body = "\n".join(_format_row(r) for r in rows)
    return header + body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reporting-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()

    rows: list[dict] = []
    for variant in VARIANTS:
        for split in SPLITS:
            predictions_path = (
                args.reporting_dir / split / "packaged" / variant / "predictions.json"
            )
            if not predictions_path.is_file():
                print(f"SKIP missing: {predictions_path}")
                continue
            records = load_predictions(predictions_path)
            for rule, k in RULE_GRID:
                row = summarize_rule(
                    records,
                    rule=rule,
                    k=k,
                    target_recall=args.target_recall,
                )
                row["variant"] = variant
                row["split"] = split
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_report(rows, args.target_recall))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

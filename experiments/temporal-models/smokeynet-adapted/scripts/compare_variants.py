"""Aggregate per-variant evaluate outputs into a Markdown comparison table.

For each ``<variant_dir>`` it reads ``predictions.json`` and ``metrics.json``
and emits a single Markdown table keyed by variant name (directory
basename). The table reports F1 at threshold 0.5, PR-AUC, ROC-AUC, and FP
count at target recalls 0.90 / 0.95 / 0.97 / 0.99.

Noise-floor interpretation is a prose rule documented in
``docs/specs/2026-04-14-performance-improvements-design.md``: a variant
must beat the baseline mean by more than the seed-to-seed spread on FP
count at target recall to count as signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_RECALLS = (0.90, 0.95, 0.97, 0.99)


def fp_at_recall(predictions: list[dict], target_recall: float) -> int | None:
    """Return the smallest FP count at which cumulative recall >= target.

    Sweeps probability thresholds from high to low and stops at the first
    threshold whose recall meets the target. Returns ``None`` when the
    prediction set has no positives or the target recall cannot be
    reached.
    """
    pairs = sorted(
        ((p["prob"], int(p["truth"])) for p in predictions),
        reverse=True,
    )
    n_pos = sum(1 for _, t in pairs if t == 1)
    if n_pos == 0:
        return None
    tp = 0
    fp = 0
    for _, truth in pairs:
        if truth == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        if recall >= target_recall:
            return fp
    return None


def summarize(variant_dirs: list[Path], output_path: Path) -> None:
    if not variant_dirs:
        raise ValueError("summarize: no variant directories provided")

    headers = [
        "variant",
        "F1 @ 0.5",
        "PR-AUC",
        "ROC-AUC",
        *[f"FP @ recall {r:.2f}" for r in TARGET_RECALLS],
    ]
    rows: list[list[str]] = []
    for variant_dir in variant_dirs:
        name = variant_dir.name
        preds_path = variant_dir / "predictions.json"
        metrics_path = variant_dir / "metrics.json"
        predictions = json.loads(preds_path.read_text())
        metrics = json.loads(metrics_path.read_text())

        row: list[str] = [
            name,
            f"{metrics.get('f1', float('nan')):.3f}",
            f"{metrics.get('pr_auc', float('nan')):.3f}",
            f"{metrics.get('roc_auc', float('nan')):.3f}",
        ]
        for target in TARGET_RECALLS:
            fp = fp_at_recall(predictions, target)
            row.append("—" if fp is None else str(fp))
        rows.append(row)

    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    prose = (
        "# Variant comparison\n\n"
        "> A variant must beat the baseline mean by more than the seed-to-seed "
        "spread on FP count at target recall to count as signal. The "
        "`train_gru`, `train_gru_seed43`, and `train_gru_seed44` rows below "
        "provide that spread.\n\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prose + "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing predictions.json + metrics.json. Repeatable.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    summarize(variant_dirs=args.variant_dir, output_path=args.output_path)


if __name__ == "__main__":
    main()

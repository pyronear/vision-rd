"""Padding ablation sweep for vit_dinov2_finetune_stabilized.

Loops a fixed grid of (pad_to_min_frames, pad_strategy) settings: for each,
packages the existing stabilized checkpoint with the override flags, evaluates
the packaged model end-to-end on the val and train splits, and aggregates the
metrics into a comparison report (markdown + CSV) plus an FPR-vs-pad plot.

Pure inference-time ablation: no retraining, no DVC DAG changes, no params.yaml
mutation. See docs/specs/2026-06-08-padding-ablation-design.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VARIANT = "vit_dinov2_finetune_stabilized"
VAL_PATCHES_DIR = Path("data/05_model_input_stabilized/val")

RUNS = [
    {"label": "baseline_pad20_sym", "pad": 20, "strategy": "symmetric"},
    {"label": "pad12_sym", "pad": 12, "strategy": "symmetric"},
    {"label": "pad8_sym", "pad": 8, "strategy": "symmetric"},
    {"label": "pad6_sym", "pad": 6, "strategy": "symmetric"},
    {"label": "pad5_sym", "pad": 5, "strategy": "symmetric"},
    {"label": "pad4_sym", "pad": 4, "strategy": "symmetric"},
    {"label": "pad2_sym", "pad": 2, "strategy": "symmetric"},
    {"label": "pad0_sym", "pad": 0, "strategy": "symmetric"},
    {"label": "pad20_uniform", "pad": 20, "strategy": "uniform"},
]

SPLITS = ("val", "train")

COLUMNS = [
    "label", "pad_to_min_frames", "pad_strategy", "split", "recall",
    "recall_ceiling", "fpr", "precision", "f1", "median_ttd_frames",
    "mean_ttd_frames", "pr_auc", "roc_auc",
]


def recall_ceiling(predictions: list[dict]) -> float | None:
    """Fraction of positive sequences that produced a surviving tube.

    A positive with ``score is None`` had no kept tube and can never fire,
    capping recall regardless of threshold. ``None`` if no positives.
    """
    positives = [p for p in predictions if p["label"] == "smoke"]
    if not positives:
        return None
    survivable = sum(1 for p in positives if p.get("score") is not None)
    return survivable / len(positives)


def summarize_run(
    *,
    label: str,
    pad: int,
    strategy: str,
    split: str,
    metrics: dict,
    predictions: list[dict],
) -> dict:
    """Flatten one run+split into a comparison row."""
    return {
        "label": label,
        "pad_to_min_frames": pad,
        "pad_strategy": strategy,
        "split": split,
        "recall": metrics["recall"],
        "recall_ceiling": recall_ceiling(predictions),
        "fpr": metrics["fpr"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "median_ttd_frames": metrics["median_ttd_frames"],
        "mean_ttd_frames": metrics["mean_ttd_frames"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
    }


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_comparison_markdown(rows: list[dict]) -> str:
    """Render rows as a GitHub-flavored markdown table."""
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = [
        "| " + " | ".join(_fmt(r.get(c)) for c in COLUMNS) + " |" for r in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"


def plot_fpr_vs_pad(rows: list[dict], output_path: Path) -> None:
    """FPR vs pad_to_min_frames, one line per split, symmetric runs only.

    Uniform runs are plotted as separate markers (different strategy).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for split in SPLITS:
        sym = sorted(
            (
                r
                for r in rows
                if r["split"] == split and r["pad_strategy"] == "symmetric"
            ),
            key=lambda r: r["pad_to_min_frames"],
        )
        if sym:
            ax.plot(
                [r["pad_to_min_frames"] for r in sym],
                [r["fpr"] for r in sym],
                marker="o",
                label=f"{split} (symmetric)",
            )
        uni = [
            r
            for r in rows
            if r["split"] == split and r["pad_strategy"] == "uniform"
        ]
        for r in uni:
            ax.scatter(
                r["pad_to_min_frames"], r["fpr"], marker="x", s=90,
                label=f"{split} (uniform, pad={r['pad_to_min_frames']})",
            )
    ax.set_xlabel("pad_to_min_frames")
    ax.set_ylabel("FPR at 0.95-recall operating point")
    ax.set_title("Padding ablation: FPR vs padding amount")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _package(run: dict, model_zip: Path) -> None:
    model_zip.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", "scripts/package_model.py",
            "--variant", VARIANT,
            "--stabilize", "true",
            "--val-patches-dir", str(VAL_PATCHES_DIR),
            "--pad-to-min-frames", str(run["pad"]),
            "--pad-strategy", run["strategy"],
            "--output", str(model_zip),
        ],
        check=True,
    )


def _evaluate(run: dict, split: str, model_zip: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", "scripts/evaluate_packaged.py",
            "--model-zip", str(model_zip),
            "--sequences-dir", f"data/01_raw/datasets/{split}",
            "--output-dir", str(out_dir),
            "--model-name", f"stabilized-{run['label']}-{split}",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/08_reporting/padding_ablation"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a run's model.zip / metrics if already present.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLITS),
        default=list(SPLITS),
        help="Which splits to evaluate this invocation. The report still "
        "aggregates any split already present on disk.",
    )
    args = parser.parse_args()

    # Compute phase: package each run, evaluate only the requested splits.
    for run in RUNS:
        run_dir = args.output_root / run["label"]
        model_zip = run_dir / "model.zip"
        if not (args.skip_existing and model_zip.exists()):
            _package(run, model_zip)
        for split in args.splits:
            split_dir = run_dir / split
            if not (args.skip_existing and (split_dir / "metrics.json").exists()):
                _evaluate(run, split, model_zip, split_dir)

    # Report phase: aggregate whatever splits exist on disk (so a baseline-only
    # train eval, or a later --splits train run, is picked up automatically).
    rows: list[dict] = []
    for run in RUNS:
        run_dir = args.output_root / run["label"]
        for split in SPLITS:
            split_dir = run_dir / split
            metrics_path = split_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            predictions = json.loads((split_dir / "predictions.json").read_text())
            rows.append(
                summarize_run(
                    label=run["label"],
                    pad=run["pad"],
                    strategy=run["strategy"],
                    split=split,
                    metrics=json.loads(metrics_path.read_text()),
                    predictions=predictions,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "comparison.md").write_text(build_comparison_markdown(rows))
    with (args.output_root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    plot_fpr_vs_pad(rows, args.output_root / "fpr_vs_pad.png")
    print(f"[sweep] wrote report to {args.output_root}", file=sys.stderr)


if __name__ == "__main__":
    main()

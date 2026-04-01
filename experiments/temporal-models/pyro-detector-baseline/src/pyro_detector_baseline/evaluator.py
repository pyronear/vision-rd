"""Sequence-level metrics computation and visualization.

Computes classification metrics (precision, recall, F1, FPR) and
time-to-detection from prediction results, and generates comparison plots
against a single-frame detection baseline.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_tracking_results(results_path: Path) -> list[dict]:
    """Load prediction results from a JSON file.

    Args:
        results_path: Path to ``tracking_results.json``.

    Returns:
        List of per-sequence result dicts.
    """
    return json.loads(results_path.read_text())


def _extract_ttd_seconds(results: list[dict]) -> list[float]:
    """Extract time-to-detection in seconds for true-positive sequences.

    Only considers sequences that are both ground-truth positive and
    predicted positive (TP), and where both timestamps are available.
    """
    ttd_seconds = []
    for r in results:
        if (
            r["is_positive_gt"]
            and r["is_positive_pred"]
            and r["confirmed_timestamp"]
            and r["first_timestamp"]
        ):
            t_first = datetime.fromisoformat(r["first_timestamp"])
            t_confirmed = datetime.fromisoformat(r["confirmed_timestamp"])
            ttd_seconds.append((t_confirmed - t_first).total_seconds())
    return ttd_seconds


def compute_metrics(results: list[dict]) -> dict:
    """Compute sequence-level classification metrics.

    Args:
        results: List of per-sequence result dicts, each containing at least
            ``is_positive_gt``, ``is_positive_pred``, ``confirmed_timestamp``,
            and ``first_timestamp``.

    Returns:
        Dict with keys: ``num_sequences``, ``num_positive_gt``,
        ``num_negative_gt``, ``tp``, ``fp``, ``fn``, ``tn``, ``precision``,
        ``recall``, ``f1``, ``fpr``, ``mean_ttd_seconds``,
        ``median_ttd_seconds``.  TTD values are ``None`` when there are no
        true positives.
    """
    tp = sum(1 for r in results if r["is_positive_gt"] and r["is_positive_pred"])
    fp = sum(1 for r in results if not r["is_positive_gt"] and r["is_positive_pred"])
    fn = sum(1 for r in results if r["is_positive_gt"] and not r["is_positive_pred"])
    tn = sum(
        1 for r in results if not r["is_positive_gt"] and not r["is_positive_pred"]
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    n_positive_gt = tp + fn
    n_negative_gt = fp + tn
    fpr = fp / n_negative_gt if n_negative_gt > 0 else 0.0

    ttd_seconds = _extract_ttd_seconds(results)
    mean_ttd = sum(ttd_seconds) / len(ttd_seconds) if ttd_seconds else None
    median_ttd = statistics.median(ttd_seconds) if ttd_seconds else None

    def row_pct(v: int, row_total: int) -> float:
        return round(v / row_total * 100, 2) if row_total > 0 else 0.0

    return {
        "num_sequences": len(results),
        "num_positive_gt": n_positive_gt,
        "num_negative_gt": n_negative_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tp_pct": row_pct(tp, n_positive_gt),
        "fp_pct": row_pct(fp, n_negative_gt),
        "fn_pct": row_pct(fn, n_positive_gt),
        "tn_pct": row_pct(tn, n_negative_gt),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "mean_ttd_seconds": (round(mean_ttd, 1) if mean_ttd is not None else None),
        "median_ttd_seconds": (
            round(median_ttd, 1) if median_ttd is not None else None
        ),
    }


def compute_single_frame_baseline(results: list[dict]) -> dict:
    """Compute baseline metrics where any detection triggers an alarm.

    Simulates a naive strategy with no temporal filtering: a sequence is
    predicted positive if it contains at least one frame where the
    predictor returned a nonzero confidence.

    Args:
        results: List of per-sequence result dicts (same format as
            :func:`compute_metrics`).

    Returns:
        Metrics dict (same keys as :func:`compute_metrics`).
    """
    baseline_results = []
    for r in results:
        baseline_results.append(
            {
                "is_positive_gt": r["is_positive_gt"],
                "is_positive_pred": r["num_detections_total"] > 0,
                "confirmed_timestamp": r["first_timestamp"],
                "first_timestamp": r["first_timestamp"],
            }
        )
    return compute_metrics(baseline_results)


def plot_confusion_matrix(metrics: dict, output_path: Path) -> None:
    """Plot a confusion matrix heatmap and save it as PNG."""
    sns.set_theme(style="whitegrid")
    cm = [[metrics["tp"], metrics["fn"]], [metrics["fp"], metrics["tn"]]]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted +", "Predicted -"],
        yticklabels=["Actual +", "Actual -"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_confusion_matrix_percentages(metrics: dict, output_path: Path) -> None:
    """Plot a confusion matrix heatmap with percentages and save it as PNG."""
    sns.set_theme(style="whitegrid")
    cm = [[metrics["tp"], metrics["fn"]], [metrics["fp"], metrics["tn"]]]
    row_totals = [sum(row) for row in cm]
    cm_pct = [
        [v / rt * 100 if rt > 0 else 0 for v in row]
        for row, rt in zip(cm, row_totals, strict=True)
    ]
    labels = [
        [f"{v / rt * 100:.1f}%" if rt > 0 else "0.0%" for v in row]
        for row, rt in zip(cm, row_totals, strict=True)
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_pct,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Predicted +", "Predicted -"],
        yticklabels=["Actual +", "Actual -"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_comparison(
    baseline_metrics: dict, predictor_metrics: dict, output_path: Path
) -> None:
    """Bar chart comparing single-frame vs predictor on precision, recall, F1."""
    sns.set_theme(style="whitegrid")
    data = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1"] * 2,
            "Score": [
                baseline_metrics["precision"],
                baseline_metrics["recall"],
                baseline_metrics["f1"],
                predictor_metrics["precision"],
                predictor_metrics["recall"],
                predictor_metrics["f1"],
            ],
            "Method": ["Single-frame"] * 3 + ["Predictor (temporal)"] * 3,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="Metric", y="Score", hue="Method", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Single-frame vs Predictor (temporal)")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_ttd_histogram(results: list[dict], output_path: Path) -> None:
    """Histogram of time-to-detection for true-positive wildfire sequences.

    Skips plotting entirely if there are no true positives.
    """
    sns.set_theme(style="whitegrid")
    ttd_seconds = _extract_ttd_seconds(results)

    if not ttd_seconds:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(ttd_seconds, bins=20, kde=True, ax=ax)
    mean_val = sum(ttd_seconds) / len(ttd_seconds)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        label=f"Mean: {mean_val:.0f}s",
    )
    ax.set_xlabel("Time to Detection (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Time to Detection Distribution (True Positives)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

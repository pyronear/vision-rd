import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def load_tracking_results(results_path: Path) -> list[dict]:
    """Load tracking results JSON."""
    return json.loads(results_path.read_text())


def compute_metrics(results: list[dict]) -> dict:
    """Compute sequence-level classification metrics."""
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

    # Time to detection (seconds from first frame to confirmation)
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

    mean_ttd = sum(ttd_seconds) / len(ttd_seconds) if ttd_seconds else None
    median_ttd = sorted(ttd_seconds)[len(ttd_seconds) // 2] if ttd_seconds else None

    return {
        "num_sequences": len(results),
        "num_positive_gt": n_positive_gt,
        "num_negative_gt": n_negative_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "mean_ttd_seconds": (round(mean_ttd, 1) if mean_ttd is not None else None),
        "median_ttd_seconds": (
            round(median_ttd, 1) if median_ttd is not None else None
        ),
    }


def compute_yolo_only_baseline(results: list[dict]) -> dict:
    """Compute baseline metrics: any YOLO detection in any frame = positive."""
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
    """Plot a confusion matrix heatmap."""
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


def plot_comparison(
    yolo_metrics: dict, tracking_metrics: dict, output_path: Path
) -> None:
    """Bar chart comparing YOLO-only vs tracking baselines."""
    sns.set_theme(style="whitegrid")
    import pandas as pd

    data = pd.DataFrame(
        {
            "Metric": ["Precision", "Recall", "F1"] * 2,
            "Score": [
                yolo_metrics["precision"],
                yolo_metrics["recall"],
                yolo_metrics["f1"],
                tracking_metrics["precision"],
                tracking_metrics["recall"],
                tracking_metrics["f1"],
            ],
            "Method": ["YOLO-only"] * 3 + ["Tracking"] * 3,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="Metric", y="Score", hue="Method", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("YOLO-only vs Tracking Baseline")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_ttd_histogram(results: list[dict], output_path: Path) -> None:
    """Histogram of time-to-detection for correctly detected WF sequences."""
    sns.set_theme(style="whitegrid")
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

"""Sequence-level metrics computation and visualization.

Computes classification metrics (precision, recall, F1, FPR) and
time-to-detection from tracking results, and generates comparison plots
against a YOLO-only baseline.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .tracker import SimpleTracker
from .types import FrameResult


def load_tracking_results(results_path: Path) -> list[dict]:
    """Load tracking results from a JSON file.

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

    Args:
        results: List of per-sequence result dicts.

    Returns:
        List of TTD values in seconds (one per qualifying TP sequence).
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


def compute_yolo_only_baseline(results: list[dict]) -> dict:
    """Compute baseline metrics where any YOLO detection triggers an alarm.

    Simulates a naive strategy with no temporal filtering: a sequence is
    predicted positive if it contains at least one detection in any frame.

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


def evaluate_tracker(
    tracker: SimpleTracker,
    all_data: list[tuple[bool, list[FrameResult]]],
    conf_thresh: float,
    max_det_area: float | None,
) -> tuple[list[dict], dict]:
    """Filter detections, run the tracker, and compute metrics.

    Args:
        tracker: Configured tracker instance.
        all_data: List of ``(ground_truth, frames)`` pairs.
        conf_thresh: Minimum detection confidence to keep.
        max_det_area: Maximum normalized detection area, or ``None``.

    Returns:
        A tuple of ``(results, metrics)`` where *results* is the list of
        per-sequence result dicts and *metrics* is the aggregated metrics dict.
    """
    results = []
    for gt, frames in all_data:
        filtered_frames = [
            FrameResult(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=[
                    d
                    for d in frame.detections
                    if d.confidence >= conf_thresh
                    and (max_det_area is None or d.w * d.h <= max_det_area)
                ],
            )
            for frame in frames
        ]

        is_alarm, _tracks, confirmed_idx, _frame_traces = tracker.process_sequence(
            filtered_frames
        )
        first_ts = filtered_frames[0].timestamp if filtered_frames else None
        confirmed_ts = (
            filtered_frames[confirmed_idx].timestamp
            if confirmed_idx is not None
            else None
        )

        results.append(
            {
                "is_positive_gt": gt,
                "is_positive_pred": is_alarm,
                "num_detections_total": sum(len(f.detections) for f in filtered_frames),
                "confirmed_timestamp": (
                    confirmed_ts.isoformat() if confirmed_ts else None
                ),
                "first_timestamp": first_ts.isoformat() if first_ts else None,
            }
        )

    return results, compute_metrics(results)


def plot_confusion_matrix(metrics: dict, output_path: Path) -> None:
    """Plot a confusion matrix heatmap and save it as PNG.

    Args:
        metrics: Metrics dict containing ``tp``, ``fp``, ``fn``, ``tn``.
        output_path: Destination file path for the PNG image.
    """
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
    """Plot a confusion matrix heatmap with percentages and save it as PNG.

    Args:
        metrics: Metrics dict containing ``tp``, ``fp``, ``fn``, ``tn``.
        output_path: Destination file path for the PNG image.
    """
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
    yolo_metrics: dict, tracking_metrics: dict, output_path: Path
) -> None:
    """Bar chart comparing YOLO-only vs tracking on precision, recall, and F1.

    Args:
        yolo_metrics: Metrics dict for the YOLO-only baseline.
        tracking_metrics: Metrics dict for the FSM tracker.
        output_path: Destination file path for the PNG image.
    """
    sns.set_theme(style="whitegrid")
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
    """Histogram of time-to-detection for true-positive wildfire sequences.

    Skips plotting entirely if there are no true positives.

    Args:
        results: List of per-sequence result dicts.
        output_path: Destination file path for the PNG image.
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

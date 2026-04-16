"""Shared matplotlib plot helpers for classification eval output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    """Render a 2x2 confusion matrix to ``output_path``.

    Args:
        matrix: 2x2 array. Rows = actual (fp, smoke). Cols = predicted.
        output_path: PNG path to write.
        title: Figure title.
        normalized: If True, format cells as percentages; else as integers.
    """
    labels = ["fp", "smoke"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    vmax = float(matrix.max()) if matrix.size else 0.0
    threshold = vmax * 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value * 100:.1f}%" if normalized else f"{int(value)}"
            color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str = "PR",
) -> None:
    """Render a precision-recall curve. Title includes AP if computable."""
    ap = float(average_precision_score(y_true, scores)) if y_true.sum() > 0 else 0.0
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"{title} (AP={ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str = "ROC",
) -> None:
    """Render a ROC curve. Title includes AUC if computable."""
    auc = (
        float(roc_auc_score(y_true, scores)) if 0 < y_true.sum() < len(y_true) else 0.0
    )
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{title} (AUC={auc:.3f})")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

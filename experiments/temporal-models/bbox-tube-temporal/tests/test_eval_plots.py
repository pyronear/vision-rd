"""Smoke tests for shared plot helpers."""

import numpy as np

from bbox_tube_temporal.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def test_plot_confusion_matrix_writes_nonempty_png(tmp_path):
    matrix = np.array([[10, 2], [3, 15]], dtype=float)
    out = tmp_path / "cm.png"

    plot_confusion_matrix(matrix, out, title="smoke test", normalized=False)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_matrix_normalized_writes_nonempty_png(tmp_path):
    matrix = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    out = tmp_path / "cm_norm.png"

    plot_confusion_matrix(matrix, out, title="smoke test (norm)", normalized=True)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pr_curve_writes_nonempty_png(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    out = tmp_path / "pr.png"

    plot_pr_curve(y_true, scores, out, title="PR")

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_roc_curve_writes_nonempty_png(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    out = tmp_path / "roc.png"

    plot_roc_curve(y_true, scores, out, title="ROC")

    assert out.exists()
    assert out.stat().st_size > 0

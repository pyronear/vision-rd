"""Unit tests for ``scripts/sweep_padding.py``'s pure helpers."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_padding.py"


def _load_script_module(alias: str):
    spec = importlib.util.spec_from_file_location(alias, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def test_recall_ceiling_counts_surviving_positives():
    mod = _load_script_module("sweep_ceiling")
    preds = [
        {"label": "smoke", "score": 1.2},
        {"label": "smoke", "score": None},  # no tube survived -> unrecoverable
        {"label": "smoke", "score": -0.3},
        {"label": "fp", "score": None},  # negatives excluded
    ]
    assert mod.recall_ceiling(preds) == 2 / 3


def test_recall_ceiling_none_when_no_positives():
    mod = _load_script_module("sweep_ceiling_empty")
    preds = [{"label": "fp", "score": 0.1}]
    assert mod.recall_ceiling(preds) is None


def test_summarize_run_pulls_metric_fields():
    mod = _load_script_module("sweep_summary")
    metrics = {
        "recall": 0.95,
        "fpr": 0.12,
        "precision": 0.8,
        "f1": 0.87,
        "median_ttd_frames": 5.0,
        "mean_ttd_frames": 6.1,
        "pr_auc": 0.91,
        "roc_auc": 0.93,
    }
    preds = [{"label": "smoke", "score": 1.0}, {"label": "smoke", "score": None}]
    row = mod.summarize_run(
        label="pad8_sym",
        pad=8,
        strategy="symmetric",
        split="val",
        metrics=metrics,
        predictions=preds,
    )
    assert row["label"] == "pad8_sym"
    assert row["pad_to_min_frames"] == 8
    assert row["split"] == "val"
    assert row["fpr"] == 0.12
    assert row["recall_ceiling"] == 0.5


def test_build_comparison_markdown_has_header_and_one_row_per_input():
    mod = _load_script_module("sweep_markdown")
    rows = [
        {
            "label": "baseline_pad20_sym", "pad_to_min_frames": 20,
            "pad_strategy": "symmetric", "split": "val", "recall": 0.95,
            "recall_ceiling": 0.97, "fpr": 0.10, "precision": 0.8, "f1": 0.87,
            "median_ttd_frames": 5.0, "mean_ttd_frames": 6.0, "pr_auc": 0.9,
            "roc_auc": 0.92,
        }
    ]
    md = mod.build_comparison_markdown(rows)
    assert "| label |" in md
    assert "baseline_pad20_sym" in md
    assert md.count("\n") >= 3  # header + separator + 1 data row

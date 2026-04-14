"""Tests for the variant comparison reporting script."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compare_variants  # noqa: E402 - test-only path setup


def _preds(pairs: list[tuple[float, int]]) -> list[dict]:
    return [
        {
            "sequence_id": f"seq_{i}",
            "truth": t,
            "prob": p,
            "predicted": int(p > 0.5),
            "correct": (p > 0.5) == bool(t),
        }
        for i, (p, t) in enumerate(pairs)
    ]


def test_fp_at_target_recall_basic():
    # 2 pos + 3 neg, sorted desc:
    # prob 0.9 pos, 0.8 neg, 0.7 pos, 0.6 neg, 0.5 neg
    # @ recall=1.0 we accept through prob 0.7 -> TP=2, FP=1
    preds = _preds([(0.9, 1), (0.8, 0), (0.7, 1), (0.6, 0), (0.5, 0)])
    assert compare_variants.fp_at_recall(preds, 1.0) == 1
    # @ recall>=0.5 we stop at first pos -> FP=0
    assert compare_variants.fp_at_recall(preds, 0.5) == 0


def test_fp_at_target_recall_all_negatives_returns_none():
    preds = _preds([(0.9, 0), (0.5, 0)])
    assert compare_variants.fp_at_recall(preds, 0.9) is None


def test_summarize_writes_markdown_with_expected_columns(tmp_path: Path):
    v1 = tmp_path / "gru"
    v1.mkdir()
    (v1 / "predictions.json").write_text(
        json.dumps(_preds([(0.9, 1), (0.8, 0), (0.7, 1), (0.6, 0)]))
    )
    (v1 / "metrics.json").write_text(
        json.dumps(
            {
                "f1": 0.9,
                "pr_auc": 0.95,
                "roc_auc": 0.96,
            }
        )
    )
    v2 = tmp_path / "convnext"
    v2.mkdir()
    (v2 / "predictions.json").write_text(
        json.dumps(_preds([(0.95, 1), (0.7, 1), (0.5, 0)]))
    )
    (v2 / "metrics.json").write_text(
        json.dumps({"f1": 0.92, "pr_auc": 0.96, "roc_auc": 0.97})
    )

    output = tmp_path / "comparison.md"
    compare_variants.summarize(
        variant_dirs=[v1, v2], output_path=output
    )

    text = output.read_text()
    assert "| variant " in text
    assert "F1 @ 0.5" in text
    assert "FP @ recall 0.90" in text
    assert "FP @ recall 0.95" in text
    assert "FP @ recall 0.97" in text
    assert "FP @ recall 0.99" in text
    assert "| gru " in text
    assert "| convnext " in text


def test_summarize_raises_on_empty(tmp_path: Path):
    output = tmp_path / "comparison.md"
    with pytest.raises(ValueError, match="no variant"):
        compare_variants.summarize(variant_dirs=[], output_path=output)

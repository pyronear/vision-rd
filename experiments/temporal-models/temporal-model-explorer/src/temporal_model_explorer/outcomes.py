"""Pure decision/outcome helpers + results filtering (no I/O, no Streamlit)."""

from __future__ import annotations

from typing import Any

import pandas as pd

ERROR_OUTCOMES = {"discarded-smoke", "kept-fp"}


def decision_from_output(is_positive: bool) -> str:
    """Map a model's is_positive verdict to keep/discard."""
    return "keep" if is_positive else "discard"


def max_probability(details: dict[str, Any] | None) -> float | None:
    """Largest calibrated probability across kept tubes, or None."""
    kept = (details or {}).get("tubes", {}).get("kept", [])
    probs = [t.get("probability") for t in kept if t.get("probability") is not None]
    return max(probs) if probs else None


def compute_outcome(decision: str, label: str) -> str:
    """Outcome of a decision vs the ground-truth label."""
    if label == "smoke":
        return "kept-smoke" if decision == "keep" else "discarded-smoke"
    if label == "fp":
        return "kept-fp" if decision == "keep" else "discarded-fp"
    return "n/a"


def filter_results(
    df: pd.DataFrame,
    *,
    model: str | None = None,
    decision: str | None = None,
    label: str | None = None,
    outcome: str | None = None,
    source: str | None = None,
    camera_name: str | None = None,
    organization_name: str | None = None,
    errors_only: bool = False,
) -> pd.DataFrame:
    """Filter a results DataFrame by any combination of column values."""
    out = df
    for col, val in (
        ("model", model),
        ("decision", decision),
        ("label", label),
        ("outcome", outcome),
        ("source", source),
        ("camera_name", camera_name),
        ("organization_name", organization_name),
    ):
        if val is not None:
            out = out[out[col] == val]
    if errors_only:
        out = out[out["outcome"].isin(ERROR_OUTCOMES)]
    return out

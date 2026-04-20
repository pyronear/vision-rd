"""Multi-model comparison and leaderboard formatting."""

import dataclasses
import json

from .types import LeaderboardEntry

# Metrics where lower is better (sorted ascending).
_LOWER_IS_BETTER = {"fpr", "mean_ttd_frames", "median_ttd_frames"}


def sort_entries(
    entries: list[LeaderboardEntry],
    primary_metric: str = "f1",
) -> list[LeaderboardEntry]:
    """Sort leaderboard entries by *primary_metric*.

    Rate metrics (precision, recall, f1) are sorted descending (higher is
    better).  FPR and TTD metrics are sorted ascending (lower is better).

    Args:
        entries: Unsorted leaderboard entries.
        primary_metric: Field name of :class:`ModelMetrics` to sort by.

    Returns:
        New list sorted by the chosen metric.
    """
    reverse = primary_metric not in _LOWER_IS_BETTER

    def key(entry: LeaderboardEntry) -> float:
        val = getattr(entry.metrics, primary_metric)
        if val is None:
            return float("inf") if not reverse else float("-inf")
        return val

    return sorted(entries, key=key, reverse=reverse)


def format_table(entries: list[LeaderboardEntry]) -> str:
    """Format leaderboard entries as an aligned plain-text table.

    Columns: Rank, Model, Precision, Recall, F1, FPR, Mean TTD, Median TTD.

    Args:
        entries: Sorted leaderboard entries.

    Returns:
        Multi-line string with aligned columns.
    """
    headers = [
        "Rank",
        "Model",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "Mean TTD (frames)",
        "Median TTD (frames)",
    ]

    rows: list[list[str]] = []
    for i, entry in enumerate(entries, start=1):
        m = entry.metrics
        rows.append(
            [
                str(i),
                m.model_name,
                f"{m.precision:.4f}",
                f"{m.recall:.4f}",
                f"{m.f1:.4f}",
                f"{m.fpr:.4f}",
                f"{m.mean_ttd_frames:.1f}" if m.mean_ttd_frames is not None else "-",
                (
                    f"{m.median_ttd_frames:.1f}"
                    if m.median_ttd_frames is not None
                    else "-"
                ),
            ]
        )

    col_widths = [
        max(len(h), *(len(row[j]) for row in rows)) for j, h in enumerate(headers)
    ]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=True))

    lines = [fmt_row(headers)]
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def to_json(entries: list[LeaderboardEntry]) -> str:
    """Serialize leaderboard entries to a JSON string.

    Returns a JSON array of objects with model_name and all metric fields.
    """
    data = [dataclasses.asdict(e.metrics) for e in entries]
    return json.dumps(data, indent=2)

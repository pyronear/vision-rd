"""Tests for temporal_model_leaderboard.leaderboard."""

import json

from temporal_model_leaderboard.leaderboard import format_table, sort_entries, to_json
from temporal_model_leaderboard.types import LeaderboardEntry, ModelMetrics


def _make_entry(
    name: str, f1: float, fpr: float = 0.1, mean_ttd: float | None = 30.0
) -> LeaderboardEntry:
    return LeaderboardEntry(
        metrics=ModelMetrics(
            model_name=name,
            num_sequences=100,
            tp=50,
            fp=5,
            fn=5,
            tn=40,
            precision=0.9,
            recall=0.9,
            f1=f1,
            fpr=fpr,
            mean_ttd_seconds=mean_ttd,
            median_ttd_seconds=mean_ttd,
        )
    )


class TestSortEntries:
    def test_sort_by_f1_descending(self) -> None:
        entries = [
            _make_entry("low", f1=0.7),
            _make_entry("high", f1=0.95),
            _make_entry("mid", f1=0.85),
        ]
        result = sort_entries(entries, "f1")
        names = [e.metrics.model_name for e in result]
        assert names == ["high", "mid", "low"]

    def test_sort_by_fpr_ascending(self) -> None:
        entries = [
            _make_entry("high_fpr", f1=0.9, fpr=0.2),
            _make_entry("low_fpr", f1=0.9, fpr=0.05),
        ]
        result = sort_entries(entries, "fpr")
        names = [e.metrics.model_name for e in result]
        assert names == ["low_fpr", "high_fpr"]

    def test_sort_by_ttd_ascending(self) -> None:
        entries = [
            _make_entry("slow", f1=0.9, mean_ttd=120.0),
            _make_entry("fast", f1=0.9, mean_ttd=30.0),
        ]
        result = sort_entries(entries, "mean_ttd_seconds")
        names = [e.metrics.model_name for e in result]
        assert names == ["fast", "slow"]

    def test_none_ttd_sorted_last(self) -> None:
        entries = [
            _make_entry("no_ttd", f1=0.9, mean_ttd=None),
            _make_entry("has_ttd", f1=0.9, mean_ttd=30.0),
        ]
        result = sort_entries(entries, "mean_ttd_seconds")
        names = [e.metrics.model_name for e in result]
        assert names == ["has_ttd", "no_ttd"]


class TestFormatTable:
    def test_contains_headers(self) -> None:
        entries = [_make_entry("model-a", f1=0.9)]
        table = format_table(entries)

        assert "Rank" in table
        assert "Model" in table
        assert "Precision" in table
        assert "F1" in table
        assert "FPR" in table

    def test_contains_model_name(self) -> None:
        entries = [_make_entry("my-model", f1=0.85)]
        table = format_table(entries)
        assert "my-model" in table

    def test_contains_values(self) -> None:
        entries = [_make_entry("m", f1=0.8500)]
        table = format_table(entries)
        assert "0.8500" in table

    def test_multiple_rows(self) -> None:
        entries = [
            _make_entry("a", f1=0.9),
            _make_entry("b", f1=0.8),
        ]
        table = format_table(entries)
        lines = table.strip().split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows

    def test_ttd_none_shows_dash(self) -> None:
        entries = [_make_entry("m", f1=0.9, mean_ttd=None)]
        table = format_table(entries)
        assert "-" in table


class TestToJson:
    def test_valid_json(self) -> None:
        entries = [_make_entry("model-a", f1=0.9)]
        result = to_json(entries)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_contains_metrics(self) -> None:
        entries = [_make_entry("model-a", f1=0.9)]
        data = json.loads(to_json(entries))

        assert data[0]["model_name"] == "model-a"
        assert data[0]["f1"] == 0.9
        assert "precision" in data[0]
        assert "fpr" in data[0]

    def test_multiple_entries(self) -> None:
        entries = [
            _make_entry("a", f1=0.9),
            _make_entry("b", f1=0.8),
        ]
        data = json.loads(to_json(entries))
        assert len(data) == 2

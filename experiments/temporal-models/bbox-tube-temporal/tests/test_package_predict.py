"""Tests for the package-time full-pipeline inference helper."""

from pathlib import Path
from unittest.mock import MagicMock

from bbox_tube_temporal import package_predict
from bbox_tube_temporal.package_predict import collect_pipeline_records


class _FakeOutput:
    def __init__(self, kept_tubes: list[dict]) -> None:
        self.details = {"tubes": {"num_candidates": len(kept_tubes), "kept": kept_tubes}}


def test_collect_pipeline_records_produces_expected_schema(
    tmp_path: Path, monkeypatch
) -> None:
    fake_sequences = [
        ("smoke", "seq_1", ["frame_a"]),
        ("smoke", "seq_2", ["frame_b"]),
        ("fp", "seq_3", ["frame_c"]),
    ]
    monkeypatch.setattr(
        package_predict,
        "_iter_labelled_sequences",
        lambda raw_dir, model: iter(fake_sequences),
    )

    fake_model = MagicMock()
    fake_model.predict.side_effect = [
        _FakeOutput([{"logit": 1.5, "start_frame": 0, "end_frame": 4, "entries": []}]),
        _FakeOutput([]),
        _FakeOutput([{"logit": -0.2, "start_frame": 0, "end_frame": 2, "entries": []}]),
    ]

    records = collect_pipeline_records(model=fake_model, raw_dir=tmp_path)

    assert [r["label"] for r in records] == ["smoke", "smoke", "fp"]
    assert [r["sequence"] for r in records] == ["seq_1", "seq_2", "seq_3"]
    assert records[0]["kept_tubes"][0]["logit"] == 1.5
    assert records[1]["kept_tubes"] == []
    assert records[2]["kept_tubes"][0]["logit"] == -0.2
    assert fake_model.predict.call_count == 3


def test_iter_labelled_sequences_uses_model_load_sequence(
    tmp_path: Path,
) -> None:
    # Build a minimal nested dataset layout.
    (tmp_path / "wildfire" / "seqA" / "images").mkdir(parents=True)
    (tmp_path / "wildfire" / "seqA" / "images" / "f1.jpg").write_bytes(b"")
    (tmp_path / "fp" / "seqB" / "images").mkdir(parents=True)
    (tmp_path / "fp" / "seqB" / "images" / "g1.jpg").write_bytes(b"")

    fake_model = MagicMock()
    fake_model.load_sequence.side_effect = lambda paths: [
        f"frame:{p.name}" for p in paths
    ]

    yielded = list(package_predict._iter_labelled_sequences(tmp_path, fake_model))

    labels = sorted((label, seq) for label, seq, _ in yielded)
    assert labels == [("fp", "seqB"), ("smoke", "seqA")]
    # Both sequences had one image; model.load_sequence was called twice.
    assert fake_model.load_sequence.call_count == 2

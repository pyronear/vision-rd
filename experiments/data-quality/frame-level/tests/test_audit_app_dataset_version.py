import json
from pathlib import Path

from data_quality_frame_level.audit_app.dataset_version import read_dataset_version


def test_returns_none_when_root_missing(tmp_path: Path):
    assert read_dataset_version(tmp_path / "nope") is None


def test_returns_none_when_source_json_missing(tmp_path: Path):
    assert read_dataset_version(tmp_path) is None


def test_returns_version_from_source_json(tmp_path: Path):
    (tmp_path / "SOURCE.json").write_text(
        json.dumps(
            {
                "pyro_dataset_version": "v4.0.0",
                "pyro_dataset_commit": "4e16c464edda7400b0ac738c4f45f8d8e50fa735",
                "refreshed_at": "2026-05-14T12:30:00+02:00",
            }
        )
    )
    assert read_dataset_version(tmp_path) == "v4.0.0"


def test_returns_none_when_source_json_malformed(tmp_path: Path):
    (tmp_path / "SOURCE.json").write_text("{not valid json")
    assert read_dataset_version(tmp_path) is None


def test_returns_none_when_version_field_absent(tmp_path: Path):
    (tmp_path / "SOURCE.json").write_text(json.dumps({"other_field": "x"}))
    assert read_dataset_version(tmp_path) is None

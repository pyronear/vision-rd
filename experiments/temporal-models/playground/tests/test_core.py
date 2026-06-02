from pathlib import Path

import pytest
from pyrocore import TemporalModelOutput

from playground.core import (
    format_summary,
    max_probability,
    resolve_frames,
    resolve_model_package,
)


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


# --- resolve_frames ---------------------------------------------------------


def test_resolve_frames_directory_globs_sorted(tmp_path):
    for name in ["002.jpg", "000.jpg", "001.png", "notes.txt"]:
        _touch(tmp_path / name)
    out = resolve_frames([str(tmp_path)])
    assert [p.name for p in out] == ["000.jpg", "001.png", "002.jpg"]


def test_resolve_frames_explicit_paths_keep_order(tmp_path):
    a = _touch(tmp_path / "b.jpg")
    b = _touch(tmp_path / "a.jpg")
    out = resolve_frames([str(a), str(b)])
    assert out == [a, b]  # given order, not sorted


def test_resolve_frames_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        resolve_frames([str(tmp_path)])


def test_resolve_frames_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_frames([str(tmp_path / "nope.jpg")])


# --- resolve_model_package --------------------------------------------------


def test_resolve_model_package_by_name(tmp_path):
    models_dir = tmp_path / "models"
    pkg = models_dir / "bbox-tube-vit-dinov2" / "model.zip"
    _touch(pkg)
    out = resolve_model_package(
        model="bbox-tube-vit-dinov2", model_package=None, models_dir=models_dir
    )
    assert out == pkg


def test_resolve_model_package_explicit_path(tmp_path):
    pkg = _touch(tmp_path / "custom.zip")
    out = resolve_model_package(
        model=None, model_package=pkg, models_dir=tmp_path / "models"
    )
    assert out == pkg


def test_resolve_model_package_requires_exactly_one(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        resolve_model_package(model=None, model_package=None, models_dir=tmp_path)


def test_resolve_model_package_unknown_name_lists_available(tmp_path):
    models_dir = tmp_path / "models"
    _touch(models_dir / "bbox-tube-vit-dinov2" / "model.zip")
    with pytest.raises(FileNotFoundError, match="bbox-tube-vit-dinov2"):
        resolve_model_package(
            model="missing", model_package=None, models_dir=models_dir
        )


# --- max_probability / format_summary ---------------------------------------


def test_max_probability_picks_largest():
    details = {"tubes": {"kept": [{"probability": 0.4}, {"probability": 0.87}]}}
    assert max_probability(details) == 0.87


def test_max_probability_none_when_absent():
    assert max_probability({"tubes": {"kept": [{"probability": None}]}}) is None
    assert max_probability({}) is None


def test_format_summary_positive_names_trigger_frame():
    frames = [Path(f"{i:03d}.jpg") for i in range(6)]
    out = TemporalModelOutput(
        is_positive=True,
        trigger_frame_index=4,
        details={"tubes": {"kept": [{"probability": 0.87}]}},
    )
    text = format_summary(out, frames, runtime_ms=412.0)
    assert "SMOKE" in text
    assert "frame 4" in text
    assert "004.jpg" in text
    assert "0.87" in text


def test_format_summary_negative_has_no_trigger():
    frames = [Path("000.jpg")]
    out = TemporalModelOutput(is_positive=False, trigger_frame_index=None, details={})
    text = format_summary(out, frames, runtime_ms=10.0)
    assert "NO SMOKE" in text
    assert "trigger" not in text.lower()

"""Tests for data_quality_sequential.registry."""

from pathlib import Path

import pytest

from data_quality_sequential.registry import MODEL_REGISTRY, load_model


def test_registry_exposes_bbox_tube_temporal() -> None:
    assert "bbox-tube-temporal" in MODEL_REGISTRY
    module_path, class_name = MODEL_REGISTRY["bbox-tube-temporal"]
    assert module_path == "bbox_tube_temporal.model"
    assert class_name == "BboxTubeTemporalModel"


def test_load_model_raises_on_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model type"):
        load_model("does-not-exist", tmp_path / "missing.zip")

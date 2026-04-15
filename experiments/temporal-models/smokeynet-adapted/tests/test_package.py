"""Tests for SmokeynetTemporalModel packaging."""

import zipfile
from pathlib import Path

import pytest
import yaml

from smokeynet_adapted.package import (
    CLASSIFIER_CKPT_FILENAME,
    CONFIG_FILENAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    YOLO_WEIGHTS_FILENAME,
    build_model_package,
)

SAMPLE_CONFIG: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 1024},
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 4,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "convnext_tiny",
        "arch": "gru",
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": False,
        "max_frames": 20,
        "pretrained": False,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.42,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


@pytest.fixture()
def dummy_yolo_weights(tmp_path: Path) -> Path:
    p = tmp_path / "yolo.pt"
    p.write_bytes(b"fake-yolo")
    return p


@pytest.fixture()
def dummy_classifier_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "classifier.ckpt"
    p.write_bytes(b"fake-classifier")
    return p


@pytest.fixture()
def built_archive(
    tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
) -> Path:
    out = tmp_path / "model.zip"
    build_model_package(
        yolo_weights_path=dummy_yolo_weights,
        classifier_ckpt_path=dummy_classifier_ckpt,
        config=SAMPLE_CONFIG,
        variant="gru_convnext_finetune",
        output_path=out,
    )
    return out


class TestBuildArchive:
    def test_output_exists(self, built_archive: Path) -> None:
        assert built_archive.exists()

    def test_is_valid_zip(self, built_archive: Path) -> None:
        assert zipfile.is_zipfile(built_archive)

    def test_contains_all_entries(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            names = set(zf.namelist())
        assert names == {
            MANIFEST_FILENAME,
            YOLO_WEIGHTS_FILENAME,
            CLASSIFIER_CKPT_FILENAME,
            CONFIG_FILENAME,
        }

    def test_yolo_weights_preserved(
        self, built_archive: Path, dummy_yolo_weights: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            assert zf.read(YOLO_WEIGHTS_FILENAME) == dummy_yolo_weights.read_bytes()

    def test_classifier_ckpt_preserved(
        self, built_archive: Path, dummy_classifier_ckpt: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            assert (
                zf.read(CLASSIFIER_CKPT_FILENAME)
                == dummy_classifier_ckpt.read_bytes()
            )


class TestManifest:
    def test_format_version(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["format_version"] == FORMAT_VERSION

    def test_variant_recorded(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["variant"] == "gru_convnext_finetune"

    def test_file_pointers(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["yolo_weights"] == YOLO_WEIGHTS_FILENAME
        assert manifest["classifier_checkpoint"] == CLASSIFIER_CKPT_FILENAME
        assert manifest["config"] == CONFIG_FILENAME


class TestConfigRoundTrip:
    def test_config_bytes_match(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            loaded = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert loaded == SAMPLE_CONFIG


class TestBuildMissingWeightsRaises:
    def test_missing_yolo(
        self, tmp_path: Path, dummy_classifier_ckpt: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            build_model_package(
                yolo_weights_path=tmp_path / "nope.pt",
                classifier_ckpt_path=dummy_classifier_ckpt,
                config=SAMPLE_CONFIG,
                variant="gru_convnext_finetune",
                output_path=tmp_path / "out.zip",
            )

    def test_missing_classifier_ckpt(
        self, tmp_path: Path, dummy_yolo_weights: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            build_model_package(
                yolo_weights_path=dummy_yolo_weights,
                classifier_ckpt_path=tmp_path / "nope.ckpt",
                config=SAMPLE_CONFIG,
                variant="gru_convnext_finetune",
                output_path=tmp_path / "out.zip",
            )

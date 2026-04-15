"""Tests for BboxTubeTemporalModel packaging."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from bbox_tube_temporal.package import (
    CLASSIFIER_CKPT_FILENAME,
    CONFIG_FILENAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    YOLO_WEIGHTS_FILENAME,
    build_model_package,
    load_model_package,
)
from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier

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
                zf.read(CLASSIFIER_CKPT_FILENAME) == dummy_classifier_ckpt.read_bytes()
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
    def test_missing_yolo(self, tmp_path: Path, dummy_classifier_ckpt: Path) -> None:
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


# Build a real tiny classifier state_dict so load_model_package can construct
# and populate a TemporalSmokeClassifier from it.
@pytest.fixture()
def real_tiny_classifier_ckpt(tmp_path: Path) -> Path:
    """A Lightning-style ckpt holding a TemporalSmokeClassifier state_dict.

    Uses backbone=resnet18 for speed in tests, regardless of the production
    variant.
    """
    model = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    # Lightning ckpt schema: torch.save({"state_dict": {...}, ...})
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": state_dict}, ckpt_path)
    return ckpt_path


@pytest.fixture()
def real_tiny_config() -> dict:
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in SAMPLE_CONFIG.items()}
    cfg["classifier"] = dict(cfg["classifier"])
    cfg["classifier"]["backbone"] = "resnet18"
    cfg["classifier"]["hidden_dim"] = 32
    return cfg


@pytest.fixture()
def real_tiny_archive(
    tmp_path: Path,
    dummy_yolo_weights: Path,
    real_tiny_classifier_ckpt: Path,
    real_tiny_config: dict,
) -> Path:
    out = tmp_path / "tiny_model.zip"
    build_model_package(
        yolo_weights_path=dummy_yolo_weights,
        classifier_ckpt_path=real_tiny_classifier_ckpt,
        config=real_tiny_config,
        variant="tiny",
        output_path=out,
    )
    return out


class TestLoadRoundtrip:
    @patch("bbox_tube_temporal.package._load_yolo")
    def test_config_passthrough(
        self,
        mock_yolo: MagicMock,
        real_tiny_archive: Path,
        tmp_path: Path,
        real_tiny_config: dict,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        pkg = load_model_package(real_tiny_archive, extract_dir=tmp_path / "ext")
        assert pkg.config == real_tiny_config

    @patch("bbox_tube_temporal.package._load_yolo")
    def test_yolo_returned(
        self, mock_yolo: MagicMock, real_tiny_archive: Path, tmp_path: Path
    ) -> None:
        sentinel = MagicMock(name="FakeYOLO")
        mock_yolo.return_value = sentinel
        pkg = load_model_package(real_tiny_archive, extract_dir=tmp_path / "ext")
        assert pkg.yolo_model is sentinel

    @patch("bbox_tube_temporal.package._load_yolo")
    def test_classifier_forward_runs(
        self,
        mock_yolo: MagicMock,
        real_tiny_archive: Path,
        tmp_path: Path,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        pkg = load_model_package(real_tiny_archive, extract_dir=tmp_path / "ext")

        patches = torch.zeros(1, 4, 3, 224, 224)
        mask = torch.tensor([[True, True, True, True]])
        with torch.no_grad():
            logit = pkg.classifier(patches, mask)
        assert logit.shape == (1,)


class TestLoadRejectsBadArchive:
    @patch("bbox_tube_temporal.package._load_yolo")
    def test_missing_manifest(self, mock_yolo: MagicMock, tmp_path: Path) -> None:
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(CONFIG_FILENAME, "infer: {}")
        with pytest.raises(KeyError):
            load_model_package(bad, extract_dir=tmp_path / "ext")

    @patch("bbox_tube_temporal.package._load_yolo")
    def test_unsupported_format_version(
        self, mock_yolo: MagicMock, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.zip"
        manifest = {
            "format_version": 99,
            "variant": "x",
            "yolo_weights": YOLO_WEIGHTS_FILENAME,
            "classifier_checkpoint": CLASSIFIER_CKPT_FILENAME,
            "config": CONFIG_FILENAME,
        }
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest))
            zf.writestr(YOLO_WEIGHTS_FILENAME, b"x")
            zf.writestr(CLASSIFIER_CKPT_FILENAME, b"x")
            zf.writestr(CONFIG_FILENAME, "{}")
        with pytest.raises(ValueError, match="format_version"):
            load_model_package(bad, extract_dir=tmp_path / "ext")

"""Tests for model packaging (build / load / round-trip)."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tracking_fsm_baseline.package import (
    CONFIG_FILENAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    WEIGHTS_FILENAME,
    build_model_package,
    load_model_package,
)
from tracking_fsm_baseline.tracker import SimpleTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PARAMS: dict = {
    "infer": {
        "confidence_threshold": 0.01,
        "iou_nms": 0.2,
        "image_size": 1024,
    },
    "pad": {
        "min_sequence_length": 10,
    },
    "track": {
        "confidence_threshold": 0.3,
        "iou_threshold": 0.1,
        "min_consecutive": 5,
        "max_detection_area": 0.05,
        "max_misses": 0,
        "use_confidence_filter": False,
        "min_mean_confidence": 0.3,
        "use_area_change_filter": False,
        "min_area_change": 1.1,
    },
}


@pytest.fixture()
def dummy_weights(tmp_path: Path) -> Path:
    """Create a small dummy weights file."""
    weights = tmp_path / "dummy.pt"
    weights.write_bytes(b"fake-yolo-weights")
    return weights


@pytest.fixture()
def built_archive(tmp_path: Path, dummy_weights: Path) -> Path:
    """Build a model archive and return its path."""
    output = tmp_path / "model.zip"
    build_model_package(dummy_weights, SAMPLE_PARAMS, output)
    return output


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------


class TestBuildCreatesValidZip:
    def test_output_exists(self, built_archive: Path) -> None:
        assert built_archive.exists()

    def test_is_valid_zip(self, built_archive: Path) -> None:
        assert zipfile.is_zipfile(built_archive)

    def test_contains_expected_entries(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            names = set(zf.namelist())
        assert names == {MANIFEST_FILENAME, WEIGHTS_FILENAME, CONFIG_FILENAME}

    def test_weights_preserved(self, built_archive: Path, dummy_weights: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            assert zf.read(WEIGHTS_FILENAME) == dummy_weights.read_bytes()


class TestBuildMissingWeightsRaises:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_model_package(
                tmp_path / "nonexistent.pt",
                SAMPLE_PARAMS,
                tmp_path / "out.zip",
            )


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestManifest:
    def test_format_version(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["format_version"] == FORMAT_VERSION

    def test_file_pointers(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["weights"] == WEIGHTS_FILENAME
        assert manifest["config"] == CONFIG_FILENAME


# ---------------------------------------------------------------------------
# Config schema tests
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_has_required_sections(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            config = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert "infer" in config
        assert "pad" in config
        assert "prefilter" in config
        assert "tracker" in config

    def test_infer_keys(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            config = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert set(config["infer"]) == {
            "confidence_threshold",
            "iou_nms",
            "image_size",
        }

    def test_prefilter_keys(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            config = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert set(config["prefilter"]) == {
            "confidence_threshold",
            "max_detection_area",
        }

    def test_tracker_keys(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            config = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert set(config["tracker"]) == {
            "iou_threshold",
            "min_consecutive",
            "max_misses",
            "use_confidence_filter",
            "min_mean_confidence",
            "use_area_change_filter",
            "min_area_change",
        }


class TestPrefilterSeparation:
    """Verify that prefilter params come from track, not infer."""

    def test_confidence_in_prefilter(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            config = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert config["prefilter"]["confidence_threshold"] == 0.3
        assert "confidence_threshold" not in config["tracker"]

    def test_max_area_in_prefilter(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            config = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert config["prefilter"]["max_detection_area"] == 0.05
        assert "max_detection_area" not in config["tracker"]


# ---------------------------------------------------------------------------
# Load / round-trip tests
# ---------------------------------------------------------------------------


class TestRoundtrip:
    @patch("tracking_fsm_baseline.package.load_model")
    def test_config_values_match(
        self, mock_load: MagicMock, built_archive: Path, tmp_path: Path
    ) -> None:
        mock_load.return_value = MagicMock(name="FakeYOLO")
        extract = tmp_path / "extract"
        pkg = load_model_package(built_archive, extract_dir=extract)
        assert pkg.infer_params["confidence_threshold"] == 0.01
        assert pkg.infer_params["iou_nms"] == 0.2
        assert pkg.infer_params["image_size"] == 1024
        assert pkg.prefilter_params["confidence_threshold"] == 0.3
        assert pkg.prefilter_params["max_detection_area"] == 0.05
        assert pkg.tracker_params["iou_threshold"] == 0.1
        assert pkg.tracker_params["min_consecutive"] == 5

    @patch("tracking_fsm_baseline.package.load_model")
    def test_model_loaded(
        self, mock_load: MagicMock, built_archive: Path, tmp_path: Path
    ) -> None:
        sentinel = MagicMock(name="FakeYOLO")
        mock_load.return_value = sentinel
        extract = tmp_path / "extract"
        pkg = load_model_package(built_archive, extract_dir=extract)
        assert pkg.model is sentinel

    @patch("tracking_fsm_baseline.package.load_model")
    def test_weights_extracted_to_dir(
        self, mock_load: MagicMock, built_archive: Path, tmp_path: Path
    ) -> None:
        mock_load.return_value = MagicMock(name="FakeYOLO")
        extract = tmp_path / "extract"
        load_model_package(built_archive, extract_dir=extract)
        assert (extract / WEIGHTS_FILENAME).exists()


class TestCreateTracker:
    @patch("tracking_fsm_baseline.package.load_model")
    def test_returns_simple_tracker(
        self, mock_load: MagicMock, built_archive: Path, tmp_path: Path
    ) -> None:
        mock_load.return_value = MagicMock(name="FakeYOLO")
        extract = tmp_path / "extract"
        pkg = load_model_package(built_archive, extract_dir=extract)
        tracker = pkg.create_tracker()
        assert isinstance(tracker, SimpleTracker)

    @patch("tracking_fsm_baseline.package.load_model")
    def test_tracker_params_match(
        self, mock_load: MagicMock, built_archive: Path, tmp_path: Path
    ) -> None:
        mock_load.return_value = MagicMock(name="FakeYOLO")
        extract = tmp_path / "extract"
        pkg = load_model_package(built_archive, extract_dir=extract)
        tracker = pkg.create_tracker()
        assert tracker.iou_threshold == 0.1
        assert tracker.min_consecutive == 5
        assert tracker.max_misses == 0
        assert tracker.use_confidence_filter is False
        assert tracker.use_area_change_filter is False


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestLoadRejectsIncompleteArchive:
    def test_missing_manifest(self, tmp_path: Path) -> None:
        bad_zip = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr(WEIGHTS_FILENAME, b"data")
            zf.writestr(CONFIG_FILENAME, "infer: {}")
        with pytest.raises(KeyError):
            load_model_package(bad_zip, extract_dir=tmp_path / "extract")

    def test_missing_weights(self, tmp_path: Path) -> None:
        bad_zip = tmp_path / "bad.zip"
        manifest = {
            "format_version": FORMAT_VERSION,
            "weights": WEIGHTS_FILENAME,
            "config": CONFIG_FILENAME,
        }
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest))
            zf.writestr(CONFIG_FILENAME, "infer: {}")
        with pytest.raises(KeyError):
            load_model_package(bad_zip, extract_dir=tmp_path / "extract")

    def test_unsupported_format_version(self, tmp_path: Path) -> None:
        bad_zip = tmp_path / "bad.zip"
        manifest = {
            "format_version": 99,
            "weights": WEIGHTS_FILENAME,
            "config": CONFIG_FILENAME,
        }
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest))
            zf.writestr(WEIGHTS_FILENAME, b"data")
            zf.writestr(CONFIG_FILENAME, "infer: {}")
        with pytest.raises(ValueError, match="format_version"):
            load_model_package(bad_zip, extract_dir=tmp_path / "extract")

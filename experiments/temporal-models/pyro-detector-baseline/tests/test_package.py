"""Tests for model packaging."""

import zipfile

import pytest
import yaml

from pyro_detector_baseline.package import (
    CONFIG_FILENAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    WEIGHTS_FILENAME,
    build_model_package,
    load_model_package,
)


@pytest.fixture()
def dummy_weights(tmp_path):
    weights = tmp_path / "best.onnx"
    weights.write_bytes(b"fake-onnx-weights")
    return weights


@pytest.fixture()
def dummy_params():
    return {
        "predict": {
            "conf_thresh": 0.35,
            "model_conf_thresh": 0.05,
            "nb_consecutive_frames": 7,
            "max_bbox_size": 0.4,
        }
    }


class TestBuildModelPackage:
    def test_creates_valid_zip(self, tmp_path, dummy_weights, dummy_params):
        out = tmp_path / "model.zip"
        build_model_package(dummy_weights, dummy_params, out)
        assert out.exists()
        assert zipfile.is_zipfile(out)

    def test_expected_entries(self, tmp_path, dummy_weights, dummy_params):
        out = tmp_path / "model.zip"
        build_model_package(dummy_weights, dummy_params, out)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert MANIFEST_FILENAME in names
        assert WEIGHTS_FILENAME in names
        assert CONFIG_FILENAME in names

    def test_manifest_content(self, tmp_path, dummy_weights, dummy_params):
        out = tmp_path / "model.zip"
        build_model_package(dummy_weights, dummy_params, out)
        with zipfile.ZipFile(out) as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["format_version"] == FORMAT_VERSION
        assert manifest["weights"] == WEIGHTS_FILENAME
        assert manifest["config"] == CONFIG_FILENAME

    def test_config_content(self, tmp_path, dummy_weights, dummy_params):
        out = tmp_path / "model.zip"
        build_model_package(dummy_weights, dummy_params, out)
        with zipfile.ZipFile(out) as zf:
            config = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert config["predict"]["conf_thresh"] == 0.35
        assert config["predict"]["nb_consecutive_frames"] == 7

    def test_weights_preserved(self, tmp_path, dummy_weights, dummy_params):
        out = tmp_path / "model.zip"
        build_model_package(dummy_weights, dummy_params, out)
        with zipfile.ZipFile(out) as zf:
            assert zf.read(WEIGHTS_FILENAME) == b"fake-onnx-weights"

    def test_missing_weights_raises(self, tmp_path, dummy_params):
        missing = tmp_path / "missing.onnx"
        out = tmp_path / "model.zip"
        with pytest.raises(FileNotFoundError):
            build_model_package(missing, dummy_params, out)


class TestLoadModelPackage:
    def _build(self, tmp_path, dummy_weights, dummy_params):
        out = tmp_path / "model.zip"
        build_model_package(dummy_weights, dummy_params, out)
        return out

    def test_roundtrip(self, tmp_path, dummy_weights, dummy_params):
        archive = self._build(tmp_path, dummy_weights, dummy_params)
        extract = tmp_path / "extract"
        pkg = load_model_package(archive, extract_dir=extract)
        assert pkg.model_path.exists()
        assert pkg.predict_params["conf_thresh"] == 0.35

    def test_missing_archive_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_package(tmp_path / "missing.zip")

    def test_missing_manifest_raises(self, tmp_path):
        archive = tmp_path / "bad.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("something.txt", "data")
        with pytest.raises(KeyError, match="manifest"):
            load_model_package(archive)

    def test_unsupported_version_raises(self, tmp_path, dummy_weights):
        archive = tmp_path / "bad.zip"
        manifest = {"format_version": 999, "weights": "w", "config": "c"}
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest))
            zf.writestr("w", b"data")
            zf.writestr("c", yaml.dump({"predict": {}}))
        with pytest.raises(ValueError, match="format_version"):
            load_model_package(archive)

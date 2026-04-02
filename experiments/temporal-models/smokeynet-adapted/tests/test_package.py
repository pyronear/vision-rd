"""Tests for smokeynet_adapted.package."""

import yaml

from smokeynet_adapted.package import FORMAT_VERSION, _build_config


class TestBuildConfig:
    def test_extracts_required_sections(self):
        params = {
            "infer": {
                "confidence_threshold": 0.01,
                "iou_nms": 0.2,
                "image_size": 1024,
            },
            "extract": {
                "roi_size": 7,
                "context_factor": 1.2,
                "max_detections_per_frame": 10,
            },
            "tubes": {"iou_threshold": 0.2, "max_misses": 2},
            "train": {
                "d_model": 512,
                "lstm_layers": 2,
                "spatial_layers": 4,
                "spatial_heads": 8,
                "classification_threshold": 0.5,
                "learning_rate": 0.0001,
                "epochs": 50,
            },
        }
        config = _build_config(params)
        assert config["infer"] == params["infer"]
        assert config["extract"] == params["extract"]
        assert config["tubes"] == params["tubes"]
        assert config["train"]["d_model"] == 512
        assert config["classification_threshold"] == 0.5
        # Should not include training-only params
        assert "learning_rate" not in config["train"]

    def test_config_is_yaml_serializable(self):
        params = {
            "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 1024},
            "extract": {
                "roi_size": 7,
                "context_factor": 1.2,
                "max_detections_per_frame": 10,
            },
            "tubes": {"iou_threshold": 0.2, "max_misses": 2},
            "train": {
                "d_model": 256,
                "lstm_layers": 1,
                "spatial_layers": 2,
                "spatial_heads": 4,
                "classification_threshold": 0.5,
            },
        }
        config = _build_config(params)
        dumped = yaml.dump(config)
        loaded = yaml.safe_load(dumped)
        assert loaded == config


class TestFormatVersion:
    def test_is_integer(self):
        assert isinstance(FORMAT_VERSION, int)
        assert FORMAT_VERSION >= 1

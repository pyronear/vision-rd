"""Smoke tests for the image-capture experiment."""

import json
from pathlib import Path


def test_cameras_json_is_valid():
    config_path = Path(__file__).parent.parent / "configs" / "cameras.json"
    config = json.loads(config_path.read_text())
    assert "cameras" in config
    assert len(config["cameras"]) > 0
    for cam in config["cameras"]:
        assert "site" in cam
        assert "ip" in cam


def test_cameras_have_unique_ips():
    config_path = Path(__file__).parent.parent / "configs" / "cameras.json"
    config = json.loads(config_path.read_text())
    ips = [cam["ip"] for cam in config["cameras"]]
    assert len(ips) == len(set(ips)), "Duplicate camera IPs found"

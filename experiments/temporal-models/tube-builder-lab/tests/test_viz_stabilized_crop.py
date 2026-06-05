"""The stabilized crop returns a square CROP_SIZE patch for any in-image window."""

from __future__ import annotations

from PIL import Image

from tube_builder_lab.viz import CROP_SIZE, stabilized_crop


def test_stabilized_crop_returns_square_patch(tmp_path):
    img_path = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 480), (123, 200, 50)).save(img_path)

    patch = stabilized_crop(img_path, (0.5, 0.5, 0.2, 0.2))

    assert patch.size == (CROP_SIZE, CROP_SIZE)

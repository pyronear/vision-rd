# tests/test_store.py

from pyrocore import Frame

from temporal_model_explorer.store import (
    FrameRef,
    SequenceMeta,
    build_frames,
    iter_sequence_dirs,
    normalize_label,
    read_meta,
    write_meta,
)

SMOKE = ["wildfire", "other_smoke"]
FP = ["other", "low_cloud"]


def _meta(key="zip_1"):
    return SequenceMeta(
        key=key,
        sequence_id="1",
        source="local_zip",
        label="smoke",
        label_detail="wildfire",
        label_source="zip_folder",
        frames=[
            FrameRef(file="images/detection_5.jpg", detection_id=5, created_at=None)
        ],
    )


def test_meta_roundtrip(tmp_path):
    d = tmp_path / "zip_1"
    write_meta(d, _meta())
    got = read_meta(d)
    assert got == _meta()
    assert (d / "meta.json").exists()


def test_iter_sequence_dirs_finds_meta_recursively(tmp_path):
    write_meta(tmp_path / "local_zip" / "zip_1", _meta("zip_1"))
    write_meta(tmp_path / "platform" / "platform_2", _meta("platform_2"))
    found = {p.name for p in iter_sequence_dirs(tmp_path)}
    assert found == {"zip_1", "platform_2"}


def test_normalize_label():
    assert normalize_label("wildfire", SMOKE, FP) == "smoke"
    assert normalize_label("other_smoke", SMOKE, FP) == "smoke"
    assert normalize_label("other", SMOKE, FP) == "fp"
    assert normalize_label("low_cloud", SMOKE, FP) == "fp"
    assert normalize_label(None, SMOKE, FP) == "unknown"
    assert normalize_label("mystery", SMOKE, FP) == "unknown"


def test_build_frames_orders_and_resolves_paths(tmp_path):
    d = tmp_path / "zip_1"
    (d / "images").mkdir(parents=True)
    (d / "images" / "detection_5.jpg").write_bytes(b"x")
    frames = build_frames(d, _meta())
    assert isinstance(frames[0], Frame)
    assert frames[0].image_path == d / "images" / "detection_5.jpg"
    assert frames[0].frame_id == "detection_5"

from pathlib import Path

from tube_builder_lab.import_sequences import import_one_by_id
from tube_builder_lab.store import read_meta


def test_import_one_orders_frames_and_writes_meta(tmp_path: Path):
    store = tmp_path / "sequences"
    detections = [
        {"id": 20, "url": "http://x/2.jpg", "created_at": "2026-05-17T10:00:30"},
        {"id": 10, "url": "http://x/1.jpg", "created_at": "2026-05-17T10:00:00"},
        {"id": 30, "url": "http://x/3.jpg", "created_at": None},  # no ts -> sorts first
    ]
    downloaded: list[str] = []

    def fake_download(url: str) -> bytes:
        downloaded.append(url)
        return b"jpeg-bytes"

    seq_dir = import_one_by_id(
        store_dir=store, sequence_id=42, detections=detections, download=fake_download
    )

    meta = read_meta(seq_dir)
    assert meta.key == "platform_42"
    assert meta.sequence_id == "42"
    # ordered by created_at ascending (None treated as empty string -> first)
    assert [f.detection_id for f in meta.frames] == [30, 10, 20]
    for f in meta.frames:
        assert (seq_dir / f.file).read_bytes() == b"jpeg-bytes"
    assert len(downloaded) == 3


def test_import_one_skips_detection_without_url(tmp_path: Path):
    store = tmp_path / "sequences"
    detections = [
        {"id": 1, "url": None, "created_at": "2026-05-17T10:00:00"},
        {"id": 2, "url": "http://x/2.jpg", "created_at": "2026-05-17T10:00:30"},
    ]
    seq_dir = import_one_by_id(
        store_dir=store, sequence_id=7, detections=detections, download=lambda u: b"x"
    )
    meta = read_meta(seq_dir)
    assert [f.detection_id for f in meta.frames] == [2]

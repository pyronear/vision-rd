import json
from pathlib import Path

from tube_builder_lab.import_sequences import copy_one_from_explorer, import_one_by_id
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


def test_copy_one_from_explorer_flattens_and_minimizes_meta(tmp_path: Path):
    # An explorer-style sequence: nested layout + a rich meta with extra fields.
    src = tmp_path / "explorer" / "sdis-77" / "cam-01" / "seq_42538"
    (src / "images").mkdir(parents=True)
    (src / "images" / "detection_5.jpg").write_bytes(b"img5")
    (src / "images" / "detection_6.jpg").write_bytes(b"img6")
    (src / "meta.json").write_text(
        json.dumps(
            {
                "key": "platform_42538",
                "sequence_id": "42538",
                "source": "platform",
                "label": "smoke",
                "camera_name": "cam-01",
                "organization_name": "sdis-77",
                "frames": [
                    {
                        "file": "images/detection_5.jpg",
                        "detection_id": 5,
                        "created_at": "2026-05-17T10:00:00",
                    },
                    {
                        "file": "images/detection_6.jpg",
                        "detection_id": 6,
                        "created_at": "2026-05-17T10:00:30",
                    },
                ],
            }
        )
    )

    lab_store = tmp_path / "lab" / "sequences"
    seq_dir = copy_one_from_explorer(lab_store=lab_store, explorer_seq_dir=src)

    # Flat layout under the lab store, keyed by the platform key.
    assert seq_dir == lab_store / "platform_42538"
    # read_meta works (minimal schema only -> no extra-key crash).
    meta = read_meta(seq_dir)
    assert meta.key == "platform_42538"
    assert meta.sequence_id == "42538"
    assert [f.detection_id for f in meta.frames] == [5, 6]
    assert (seq_dir / "images" / "detection_5.jpg").read_bytes() == b"img5"
    assert (seq_dir / "images" / "detection_6.jpg").read_bytes() == b"img6"

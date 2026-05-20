# tests/test_import_platform.py
from datetime import date

import temporal_model_explorer.import_platform as ip
from temporal_model_explorer.store import read_meta

SMOKE = ["wildfire", "other_smoke"]
FP = ["other"]


def test_import_platform_writes_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ip.platform_api,
        "list_sequences_for_date",
        lambda ep, tok, day, limit, offset: [
            {
                "id": 43392,
                "camera_id": 7,
                "is_wildfire": "other_smoke",
                "started_at": "2026-05-19T14:10:01Z",
            }
        ],
    )
    monkeypatch.setattr(
        ip.platform_api,
        "list_sequence_detections",
        lambda ep, tok, sid, limit, desc: [
            {"id": 2, "url": "http://img/2", "created_at": "2026-05-19T14:10:31Z"},
            {"id": 1, "url": "http://img/1", "created_at": "2026-05-19T14:10:01Z"},
        ],
    )
    camera_index = {7: {"id": 7, "name": "cam-7", "organization_id": 3}}
    n = ip.import_platform(
        "https://x",
        "tok",
        tmp_path,
        date(2026, 5, 19),
        date(2026, 5, 19),
        detections_limit=5,
        smoke_values=SMOKE,
        fp_values=FP,
        camera_index=camera_index,
        org_index={3: "demo"},
        download=lambda url: f"BYTES:{url}".encode(),
    )
    assert n == 1
    # organized on disk by org/camera: <org>/<camera>/seq_<id>/
    seq_dir = tmp_path / "demo" / "cam-7" / "seq_43392"
    meta = read_meta(seq_dir)
    assert meta.key == "platform_43392"
    assert meta.label == "smoke" and meta.label_detail == "other_smoke"
    assert meta.camera_name == "cam-7" and meta.organization_id == 3
    assert meta.organization_name == "demo"
    # frames ordered by created_at ascending (detection 1 then 2)
    assert [f.detection_id for f in meta.frames] == [1, 2]
    assert (
        seq_dir / "images" / "detection_1.jpg"
    ).read_bytes() == b"BYTES:http://img/1"


def test_import_platform_org_slug_fallback_without_org_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ip.platform_api,
        "list_sequences_for_date",
        lambda ep, tok, day, limit, offset: [
            {"id": 5, "camera_id": 7, "is_wildfire": "other"}
        ],
    )
    monkeypatch.setattr(
        ip.platform_api,
        "list_sequence_detections",
        lambda ep, tok, sid, limit, desc: [
            {"id": 1, "url": "http://img/1", "created_at": "t"}
        ],
    )
    ip.import_platform(
        "https://x",
        "tok",
        tmp_path,
        date(2026, 5, 19),
        date(2026, 5, 19),
        detections_limit=5,
        smoke_values=SMOKE,
        fp_values=FP,
        camera_index={7: {"id": 7, "name": "cam-7", "organization_id": 3}},
        download=lambda url: b"x",
    )
    # no org_index -> org_<id>/<camera>/seq_<id>, organization_name stays None
    meta = read_meta(tmp_path / "org_3" / "cam-7" / "seq_5")
    assert meta.organization_id == 3 and meta.organization_name is None


def test_camera_filter_excludes_other_cameras(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ip.platform_api,
        "list_sequences_for_date",
        lambda ep, tok, day, limit, offset: [
            {"id": 1, "camera_id": 99, "is_wildfire": "other"}
        ],
    )
    n = ip.import_platform(
        "https://x",
        "tok",
        tmp_path,
        date(2026, 5, 19),
        date(2026, 5, 19),
        detections_limit=5,
        smoke_values=SMOKE,
        fp_values=FP,
        camera_ids={7},
        camera_index={},
        download=lambda url: b"x",
    )
    assert n == 0

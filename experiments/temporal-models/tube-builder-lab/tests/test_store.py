from pathlib import Path

from tube_builder_lab.store import (
    FrameRef,
    SequenceMeta,
    build_frames,
    iter_sequence_dirs,
    read_meta,
    seq_dir_for_key,
    write_meta,
)


def test_meta_roundtrip_and_lookup(tmp_path: Path):
    store = tmp_path / "sequences"
    meta = SequenceMeta(
        key="platform_42",
        sequence_id="42",
        frames=[
            FrameRef(
                file="images/a.jpg", detection_id=1, created_at="2026-05-17T10:00:00"
            ),
            FrameRef(
                file="images/b.jpg", detection_id=2, created_at="2026-05-17T10:00:30"
            ),
        ],
    )
    seq_dir = store / "platform_42"
    write_meta(seq_dir, meta)

    assert [d.name for d in iter_sequence_dirs(store)] == ["platform_42"]
    assert seq_dir_for_key(store, "platform_42") == seq_dir
    assert seq_dir_for_key(store, "nope") is None

    got = read_meta(seq_dir)
    assert got == meta

    frames = build_frames(seq_dir, got)
    assert [f.frame_id for f in frames] == ["a", "b"]
    assert frames[0].image_path == seq_dir / "images/a.jpg"
    assert frames[1].timestamp is not None

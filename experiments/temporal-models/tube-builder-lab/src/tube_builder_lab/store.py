"""Local sequence store (flat `<key>/` layout): meta IO + Frame helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from pyrocore import Frame

META_FILENAME = "meta.json"


@dataclass
class FrameRef:
    file: str  # relative to the sequence dir, e.g. "images/detection_5.jpg"
    detection_id: int | None = None
    created_at: str | None = None  # ISO timestamp


@dataclass
class SequenceMeta:
    key: str
    sequence_id: str
    frames: list[FrameRef] = field(default_factory=list)


def write_meta(seq_dir: Path, meta: SequenceMeta) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / META_FILENAME).write_text(json.dumps(asdict(meta), indent=2))


def read_meta(seq_dir: Path) -> SequenceMeta:
    payload = json.loads((seq_dir / META_FILENAME).read_text())
    frames = [FrameRef(**f) for f in payload.pop("frames", [])]
    return SequenceMeta(frames=frames, **payload)


def iter_sequence_dirs(store_dir: Path) -> Iterator[Path]:
    """Yield every directory under ``store_dir`` containing a meta.json."""
    if not store_dir.exists():
        return
    for meta_path in sorted(store_dir.rglob(META_FILENAME)):
        yield meta_path.parent


def seq_dir_for_key(store_dir: Path, key: str) -> Path | None:
    """Resolve a sequence dir by its meta key (flat layout = store/<key>)."""
    direct = store_dir / key
    if (direct / META_FILENAME).exists():
        return direct
    for seq_dir in iter_sequence_dirs(store_dir):
        if read_meta(seq_dir).key == key:
            return seq_dir
    return None


def build_frames(seq_dir: Path, meta: SequenceMeta) -> list[Frame]:
    """Ordered pyrocore Frames; meta order is the time axis."""
    frames: list[Frame] = []
    for ref in meta.frames:
        ts = None
        if ref.created_at:
            try:
                ts = datetime.fromisoformat(ref.created_at.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        frames.append(
            Frame(
                frame_id=Path(ref.file).stem,
                image_path=seq_dir / ref.file,
                timestamp=ts,
            )
        )
    return frames

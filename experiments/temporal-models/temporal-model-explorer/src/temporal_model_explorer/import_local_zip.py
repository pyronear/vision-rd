"""Import the annotated `seq_annotation_done_by_label` zip into the sequence store.

Zip layout: <root>/<category>[/<detail>]/seq_<id>/images/detection_<id>.jpg
  category: "smoke" | "fp" | "unlabeled"; detail: subfolder (wildfire, tree, …).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .store import FrameRef, SequenceMeta, write_meta

_DET_RE = re.compile(r"detection_(\d+)")


def label_from_parts(parts: tuple[str, ...]) -> tuple[str, str | None]:
    """Map category path components (zip root → seq dir) to (label, detail)."""
    if not parts:
        return "unknown", None
    top = parts[0]
    detail = parts[1] if len(parts) > 1 else None
    if top == "smoke":
        return "smoke", detail
    if top == "fp":
        return "fp", detail
    return "unknown", None


def _detection_id(file_name: str) -> int | None:
    m = _DET_RE.search(file_name)
    return int(m.group(1)) if m else None


def import_zip(zip_path: Path, store_dir: Path) -> int:
    """Extract image frames + write meta.json per sequence.

    Returns the number of sequences imported.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict] = {}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            if (
                "__MACOSX" in name
                or name.endswith("/")
                or Path(name).name == ".DS_Store"
            ):
                continue
            parts = Path(name).parts
            if "images" not in parts:
                continue
            idx = parts.index("images")
            if idx == 0 or not parts[idx - 1].startswith("seq_"):
                continue
            seq_dirname = parts[idx - 1]
            category_parts = parts[1 : idx - 1]  # drop zip root at parts[0]
            label, detail = label_from_parts(tuple(category_parts))
            entry = grouped.setdefault(
                seq_dirname, {"label": label, "detail": detail, "files": []}
            )
            entry["files"].append((name, parts[-1]))

        count = 0
        for seq_dirname, entry in grouped.items():
            seq_id = seq_dirname.removeprefix("seq_")
            key = f"zip_{seq_id}"
            out_dir = store_dir / key
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            frames: list[FrameRef] = []
            for src_name, file_name in sorted(entry["files"], key=lambda t: t[1]):
                (out_dir / "images" / file_name).write_bytes(zf.read(src_name))
                frames.append(
                    FrameRef(
                        file=f"images/{file_name}",
                        detection_id=_detection_id(file_name),
                    )
                )
            write_meta(
                out_dir,
                SequenceMeta(
                    key=key,
                    sequence_id=seq_id,
                    source="local_zip",
                    label=entry["label"],
                    label_detail=entry["detail"],
                    label_source="zip_folder",
                    frames=frames,
                ),
            )
            count += 1
    return count

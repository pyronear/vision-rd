"""Read-only viewer for exported YOLO labels under ``data/10_export/``.

Routes:
  GET /api/contexts             — exported (model, split) pairs with frame counts
  GET /api/frames?model&split   — sorted stems present in the export
  GET /api/sample?model&split&stem — parsed YOLO bboxes for the stem
  GET /image?split&stem         — JPEG bytes from data/01_raw/datasets/<split>/images
  GET /                         — static index.html
"""

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from data_quality_frame_level.audit_app.sequence import (
    assign_temporal_sequences,
    parse_stem,
)


@dataclass(frozen=True)
class ExportContext:
    model: str
    split: str
    labels_dir: Path
    images_dir: Path


def discover_contexts(repo_root: Path) -> list[ExportContext]:
    export_root = repo_root / "data" / "10_export"
    datasets_root = repo_root / "data" / "01_raw" / "datasets"
    if not export_root.is_dir():
        return []
    found: list[ExportContext] = []
    for model_dir in sorted(p for p in export_root.iterdir() if p.is_dir()):
        for split_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            labels_dir = split_dir / "labels"
            images_dir = datasets_root / split_dir.name / "images"
            if labels_dir.is_dir():
                found.append(
                    ExportContext(
                        model=model_dir.name,
                        split=split_dir.name,
                        labels_dir=labels_dir,
                        images_dir=images_dir,
                    )
                )
    return found


def list_stems(labels_dir: Path) -> list[str]:
    return sorted(p.stem for p in labels_dir.glob("*.txt"))


def parse_yolo_label(path: Path) -> list[dict]:
    bboxes: list[dict] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"malformed YOLO line in {path}: {raw!r}")
        class_id = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:])
        bboxes.append({"class_id": class_id, "cx": cx, "cy": cy, "w": w, "h": h})
    return bboxes


def create_app(*, contexts: list[ExportContext]) -> FastAPI:
    by_key = {(c.model, c.split): c for c in contexts}
    images_by_split = {c.split: c.images_dir for c in contexts}

    def _ctx(model: str, split: str) -> ExportContext:
        key = (model, split)
        if key not in by_key:
            raise HTTPException(404, f"unknown context: {key}")
        return by_key[key]

    app = FastAPI()

    @app.get("/api/contexts")
    def get_contexts() -> dict:
        items = [
            {
                "model": c.model,
                "split": c.split,
                "count": len(list(c.labels_dir.glob("*.txt"))),
            }
            for c in contexts
        ]
        return {"items": items}

    @app.get("/api/frames")
    def get_frames(model: str, split: str) -> dict:
        ctx = _ctx(model, split)
        stems = list_stems(ctx.labels_dir)
        seq_by_stem = assign_temporal_sequences(stems)
        groups: dict[str, list[tuple[str, str]]] = {}
        for stem in stems:
            _, ts = parse_stem(stem)
            groups.setdefault(seq_by_stem[stem], []).append((ts, stem))
        items = []
        for seq_id, frames in groups.items():
            frames.sort()
            items.append(
                {
                    "sequence_id": seq_id,
                    "stems": [s for _, s in frames],
                }
            )
        items.sort(key=lambda g: g["sequence_id"])
        return {"sequences": items}

    @app.get("/api/sample")
    def get_sample(model: str, split: str, stem: str) -> dict:
        ctx = _ctx(model, split)
        label_path = ctx.labels_dir / f"{stem}.txt"
        if not label_path.is_file():
            raise HTTPException(404, f"unknown stem: {stem}")
        image_path = ctx.images_dir / f"{stem}.jpg"
        return {
            "stem": stem,
            "bboxes": parse_yolo_label(label_path),
            "image_available": image_path.is_file(),
        }

    @app.get("/image")
    def get_image(split: str, stem: str) -> FileResponse:
        images_dir = images_by_split.get(split)
        if images_dir is None:
            raise HTTPException(404, f"unknown split: {split}")
        path = images_dir / f"{stem}.jpg"
        if not path.is_file():
            raise HTTPException(404, f"missing image: {stem}")
        return FileResponse(path, media_type="image/jpeg")

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app

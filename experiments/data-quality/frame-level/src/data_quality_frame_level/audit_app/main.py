"""FastAPI app for the frame-level review workflow.

Routes:
  GET  /api/contexts                                 — available models + splits
  GET  /api/queue?model&split&view&conf&iou&review_conf  — ordered queue
  GET  /api/sample?model&split&stem&conf&iou&review_conf — layers + neighbors
  POST /api/sample?model&split  (body: SaveBody)     — save verified GT
  POST /api/export?model&split&conf&iou&review_conf  — write data/10_export
  GET  /image?model&split&stem                       — JPEG bytes
  GET  /                                             — static index.html
"""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data_quality_frame_level.audit_app.dataset_version import read_dataset_version
from data_quality_frame_level.audit_app.export_runner import export_one
from data_quality_frame_level.audit_app.matching import evaluate_frame
from data_quality_frame_level.audit_app.persistence import dvc_warning_for_review
from data_quality_frame_level.audit_app.queue import build_queue
from data_quality_frame_level.audit_app.sequence import parse_stem
from data_quality_frame_level.audit_app.state import AppState, Paths
from data_quality_frame_level.dataset import BBox


class BBoxModel(BaseModel):
    class_id: int = 0
    cx: float
    cy: float
    w: float
    h: float


class SaveBody(BaseModel):
    stem: str
    status: str = Field(..., pattern="^(reviewed|unclear)$")
    bboxes: list[BBoxModel]
    spurious_originals: list[BBoxModel] = Field(default_factory=list)
    reviewer: str | None = None
    note: str | None = None


def create_app(
    *,
    contexts: dict[tuple[str, str], Paths],
    models: list[str],
    splits: list[str],
    repo_root: Path,
) -> FastAPI:
    cache: dict[tuple[str, str], AppState] = {}

    def _state(model: str, split: str) -> AppState:
        key = (model, split)
        if key not in contexts:
            raise HTTPException(404, f"unknown context: {key}")
        if key not in cache:
            cache[key] = AppState.load(model=model, split=split, paths=contexts[key])
        return cache[key]

    app = FastAPI()

    @app.get("/api/contexts")
    def get_contexts() -> dict:
        warnings: list[dict] = []
        for (m, s), paths in contexts.items():
            w = dvc_warning_for_review(paths.review_path)
            if w is not None:
                warnings.append({**w, "model": m, "split": s})
        return {
            "models": models,
            "splits": splits,
            "dvc_warnings": warnings,
            "dataset_version": read_dataset_version(
                repo_root / "data" / "01_raw" / "datasets"
            ),
        }

    @app.get("/api/queue")
    def get_queue(
        model: str,
        split: str,
        view: str,
        conf: float,
        iou: float,
        review_conf: float,
    ) -> dict:
        s = _state(model, split)
        flagged = build_queue(
            predictions=s.predictions,
            gt=s.gt,
            sequence_id_by_stem=s.sequence_id_by_stem,
            review_status={k: v.status for k, v in s.review.samples.items()},
            view=view,
            conf_thresh=conf,
            iou_thresh=iou,
            review_conf_thresh=review_conf,
        )
        flagged_stems = {it.stem for it in flagged}
        flagged_seq_ids = {it.sequence_id for it in flagged}
        seq_max: dict[str, float] = {}
        for it in flagged:
            if it.severity > seq_max.get(it.sequence_id, 0.0):
                seq_max[it.sequence_id] = it.severity
        expanded = [asdict(i) for i in flagged]
        for stem in s.gt.keys() | s.predictions.keys():
            if stem in flagged_stems:
                continue
            seq_id = s.sequence_id_by_stem[stem]
            if seq_id not in flagged_seq_ids:
                continue
            _, ts = parse_stem(stem)
            sample = s.review.samples.get(stem)
            expanded.append(
                {
                    "stem": stem,
                    "sequence_id": seq_id,
                    "timestamp": ts,
                    "kind": "none",
                    "severity": 0.0,
                    "status": sample.status if sample else None,
                }
            )
        expanded.sort(
            key=lambda i: (-seq_max[i["sequence_id"]], i["sequence_id"], i["timestamp"])
        )
        return {"items": expanded}

    @app.get("/api/sample")
    def get_sample(
        model: str,
        split: str,
        stem: str,
        conf: float,
        iou: float,
        review_conf: float,
    ) -> dict:
        s = _state(model, split)
        if stem not in s.gt and stem not in s.predictions:
            raise HTTPException(404, f"unknown stem: {stem}")
        gt = s.gt.get(stem, [])
        preds = [p for p in s.predictions.get(stem, []) if p.conf >= conf]
        ev = evaluate_frame(gt=gt, predictions=preds, iou_thresh=iou)
        sample = s.review.samples.get(stem)
        seq_id = s.sequence_id_by_stem[stem]
        _, ts = parse_stem(stem)
        neighbors = sorted(
            (
                {"stem": st, "timestamp": parse_stem(st)[1]}
                for st in s.gt.keys() | s.predictions.keys()
                if s.sequence_id_by_stem[st] == seq_id
            ),
            key=lambda d: d["timestamp"],
        )
        return {
            "stem": stem,
            "sequence_id": seq_id,
            "timestamp": ts,
            "original_gt": [
                {**asdict(b), "status": st}
                for b, st in zip(gt, ev.gt_status, strict=True)
            ],
            "predictions": [
                {**asdict(p), "status": st}
                for p, st in zip(preds, ev.pred_status, strict=True)
            ],
            "verified_gt": [asdict(b) for b in (sample.bboxes if sample else [])],
            "spurious_originals": [
                asdict(b) for b in (sample.spurious_originals if sample else [])
            ],
            "status": sample.status if sample else None,
            "reviewer": sample.reviewer if sample else None,
            "note": sample.note if sample else None,
            "reviewed_at": sample.reviewed_at if sample else None,
            "sequence_neighbors": neighbors,
        }

    @app.post("/api/sample")
    def save_sample(model: str, split: str, body: SaveBody) -> dict:
        s = _state(model, split)
        bboxes = [
            BBox(class_id=b.class_id, cx=b.cx, cy=b.cy, w=b.w, h=b.h)
            for b in body.bboxes
        ]
        spurious = [
            BBox(class_id=b.class_id, cx=b.cx, cy=b.cy, w=b.w, h=b.h)
            for b in body.spurious_originals
        ]
        sample = s.save_sample(
            stem=body.stem,
            status=body.status,
            bboxes=bboxes,
            spurious_originals=spurious,
            reviewer=body.reviewer,
            note=body.note,
        )
        return {"saved_at": sample.reviewed_at}

    @app.delete("/api/sample")
    def delete_sample(model: str, split: str, stem: str) -> dict:
        s = _state(model, split)
        s.delete_sample(stem=stem)
        return {"ok": True}

    @app.post("/api/export")
    def post_export(
        model: str,
        split: str,
        conf: float,
        iou: float,
        review_conf: float,
    ) -> dict:
        if (model, split) not in contexts:
            raise HTTPException(404, f"unknown context: ({model}, {split})")
        manifest = export_one(
            repo_root=repo_root,
            model=model,
            split=split,
            conf=conf,
            iou=iou,
            review_conf=review_conf,
        )
        if manifest is None:
            raise HTTPException(
                400,
                "nothing to export — review.json or predictions.json missing",
            )
        return manifest

    @app.get("/image")
    def get_image(model: str, split: str, stem: str) -> FileResponse:
        s = _state(model, split)
        path = s.paths.split_dir / "images" / f"{stem}.jpg"
        if not path.is_file():
            raise HTTPException(404, f"missing image: {stem}")
        return FileResponse(path, media_type="image/jpeg")

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app

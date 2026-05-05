"""FastAPI app for the frame-level review workflow.

Routes:
  GET  /api/contexts                                 — available models + splits
  GET  /api/queue?model&split&view&conf&iou&review_conf  — ordered queue
  GET  /api/sample?model&split&stem&conf&iou&review_conf — layers + neighbors
  POST /api/sample?model&split  (body: SaveBody)     — save corrected GT
  GET  /image?model&split&stem                       — JPEG bytes
  GET  /                                             — static index.html
"""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.matching import evaluate_frame
from data_quality_frame_level.review_app.queue import build_queue
from data_quality_frame_level.review_app.sequence import parse_stem
from data_quality_frame_level.review_app.state import AppState, Paths


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
    reviewer: str | None = None
    note: str | None = None


def create_app(
    *,
    contexts: dict[tuple[str, str], Paths],
    models: list[str],
    splits: list[str],
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
        return {"models": models, "splits": splits}

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
        items = build_queue(
            predictions=s.predictions,
            gt=s.gt,
            review_status={k: v.status for k, v in s.review.samples.items()},
            view=view,
            conf_thresh=conf,
            iou_thresh=iou,
            review_conf_thresh=review_conf,
        )
        return {"items": [asdict(i) for i in items]}

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
        seq_id, ts = parse_stem(stem)
        neighbors = sorted(
            (
                {"stem": st, "timestamp": parse_stem(st)[1]}
                for st in s.gt.keys() | s.predictions.keys()
                if parse_stem(st)[0] == seq_id
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
            "corrected_gt": [asdict(b) for b in (sample.bboxes if sample else [])],
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
        sample = s.save_sample(
            stem=body.stem,
            status=body.status,
            bboxes=bboxes,
            reviewer=body.reviewer,
            note=body.note,
        )
        return {"saved_at": sample.reviewed_at}

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

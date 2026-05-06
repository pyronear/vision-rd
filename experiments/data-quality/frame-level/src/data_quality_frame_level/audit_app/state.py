"""In-memory state for one ``(model, split)`` context.

Holds the universe of frames (predictions + GT) plus the review state.
Constructed once per context and cached by the FastAPI layer.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from data_quality_frame_level.audit_app.persistence import (
    ReviewState,
    SampleReview,
    read_review_state,
    write_review_state,
)
from data_quality_frame_level.audit_app.sequence import assign_temporal_sequences
from data_quality_frame_level.audit_app.types import Prediction
from data_quality_frame_level.dataset import BBox, iter_frames


@dataclass(frozen=True)
class Paths:
    split_dir: Path
    predictions_path: Path
    review_path: Path


def _load_predictions(path: Path) -> dict[str, list[Prediction]]:
    raw = json.loads(path.read_text())
    out: dict[str, list[Prediction]] = {}
    for stem, frame in raw["frames"].items():
        out[stem] = [
            Prediction(
                class_id=int(d["class_id"]),
                cx=float(d["cx"]),
                cy=float(d["cy"]),
                w=float(d["w"]),
                h=float(d["h"]),
                conf=float(d["conf"]),
            )
            for d in frame["predictions"]
        ]
    return out


def _load_gt(split_dir: Path) -> dict[str, list[BBox]]:
    return {f.stem: f.gt_bboxes for f in iter_frames(split_dir)}


@dataclass
class AppState:
    model: str
    split: str
    paths: Paths
    predictions: dict[str, list[Prediction]]
    gt: dict[str, list[BBox]]
    review: ReviewState
    sequence_id_by_stem: dict[str, str]

    @classmethod
    def load(cls, *, model: str, split: str, paths: Paths) -> "AppState":
        predictions = _load_predictions(paths.predictions_path)
        gt = _load_gt(paths.split_dir)
        all_stems = predictions.keys() | gt.keys()
        return cls(
            model=model,
            split=split,
            paths=paths,
            predictions=predictions,
            gt=gt,
            review=read_review_state(paths.review_path, model=model, split=split),
            sequence_id_by_stem=assign_temporal_sequences(all_stems),
        )

    def save_sample(
        self,
        *,
        stem: str,
        status: str,
        bboxes: list[BBox],
        spurious_originals: list[BBox],
        reviewer: str | None,
        note: str | None,
    ) -> SampleReview:
        sample = SampleReview(
            status=status,
            bboxes=list(bboxes),
            spurious_originals=list(spurious_originals),
            reviewer=reviewer,
            note=note,
            reviewed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self.review.samples[stem] = sample
        write_review_state(self.paths.review_path, self.review)
        return sample

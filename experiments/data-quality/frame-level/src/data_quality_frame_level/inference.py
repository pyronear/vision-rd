"""Thin wrapper around ultralytics YOLO for single-frame batched inference.

Returns per-image lists of predictions in the same normalized-center
format as :class:`data_quality_frame_level.dataset.BBox`, so downstream
code can apply the same YOLO-center -> FiftyOne-top-left conversion to
both GT and predicted bboxes.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO


@dataclass(frozen=True)
class PredBBox:
    """One YOLO prediction in normalized center form + confidence."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float


def load_model(model_path: Path) -> YOLO:
    """Load a YOLO model from a local ``.pt`` file.

    Device placement is handled by :func:`predict_images` via
    ``YOLO.predict(device=...)`` — no explicit ``.to(device)`` here
    because torch rejects ultralytics' bare numeric device strings
    (e.g. ``"0"``) while ultralytics accepts them and maps them to
    ``cuda:0`` internally.
    """
    return YOLO(str(model_path))


def predict_images(
    model: YOLO,
    image_paths: Iterable[Path],
    conf_thresh: float,
    device: str = "0",
) -> list[list[PredBBox]]:
    """Run YOLO on a sequence of images; return per-image predictions.

    Preserves input order. Uses ultralytics' ``xywhn`` (normalized
    center) output directly so coords line up with :class:`BBox`.
    """
    image_paths = list(image_paths)
    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=conf_thresh,
        device=device,
        verbose=False,
        stream=False,
    )
    output: list[list[PredBBox]] = []
    for result in results:
        preds: list[PredBBox] = []
        if result.boxes is None:
            output.append(preds)
            continue
        for box in result.boxes:
            cx, cy, w, h = box.xywhn[0].tolist()
            preds.append(
                PredBBox(
                    class_id=int(box.cls.item()),
                    cx=float(cx),
                    cy=float(cy),
                    w=float(w),
                    h=float(h),
                    conf=float(box.conf.item()),
                )
            )
        output.append(preds)
    return output

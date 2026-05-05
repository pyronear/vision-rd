"""Plain dataclasses for the review app.

Defined locally so the review_app package has no dependency on
ultralytics / YOLO runtime — that lives in
:mod:`data_quality_frame_level.inference` and is only needed to
*produce* predictions, not to *consume* them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float

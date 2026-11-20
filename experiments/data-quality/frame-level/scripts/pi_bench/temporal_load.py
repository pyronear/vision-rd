"""Every PERIOD s: one 3-tube job on 4 frames (boxes move to force cache misses)."""

import json
import sys
import time
from pathlib import Path

from temporal_model.core import Detection, FrameDetections
from temporal_model.core.onnx_model import OnnxTemporalModel

cycles, period = int(sys.argv[1]), float(sys.argv[2])
m = OnnxTemporalModel.from_package(
    Path("tm_v061/model_onnx.zip"), embedding_cache_size=4096
)
paths = sorted(Path(sys.argv[3]).glob("*.jpg"))[:4]
for c in range(cycles):
    start = time.time()
    frames = m.load_sequence(paths)
    boxes = [
        Detection(
            class_id=0,
            cx=0.15 + 0.3 * k + 0.0007 * c,
            cy=0.3 + 0.001 * c,
            w=0.05,
            h=0.03,
            confidence=0.9,
        )
        for k in range(3)
    ]
    dets = {
        f.frame_id: FrameDetections(
            frame_idx=i, frame_id=f.frame_id, timestamp=f.timestamp, detections=boxes
        )
        for i, f in enumerate(frames)
    }
    out = m.predict(frames, frame_detections=dets)
    print(
        json.dumps(
            {
                "cycle": c,
                "job_s": round(time.time() - start, 2),
                "tubes": len(out.details["tubes"]["kept"]),
            }
        ),
        flush=True,
    )
    time.sleep(max(period - (time.time() - start), 0))

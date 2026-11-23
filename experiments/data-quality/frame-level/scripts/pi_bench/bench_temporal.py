"""Time temporal jobs as pyro_temporal_api runs them (ONNX, embedding cache)."""

import json
import statistics
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from temporal_model.core import Detection, FrameDetections
from temporal_model.core.onnx_model import OnnxTemporalModel

pkg = Path(
    hf_hub_download(
        "pyronear/temporal-model",
        "model_onnx.zip",
        revision="v0.6.1",
        local_dir="tm_v061",
    )
)
paths = sorted(Path(sys.argv[1]).glob("*.jpg"))[:10]


def job(model, n, n_boxes):
    frames = model.load_sequence(paths[:n])
    boxes = [
        Detection(class_id=0, cx=0.2 + 0.25 * k, cy=0.4, w=0.05, h=0.03, confidence=0.9)
        for k in range(n_boxes)
    ]
    dets = {
        f.frame_id: FrameDetections(
            frame_idx=i, frame_id=f.frame_id, timestamp=f.timestamp, detections=boxes
        )
        for i, f in enumerate(frames)
    }
    t = time.perf_counter()
    out = model.predict(frames, frame_detections=dets)
    return round(1000 * (time.perf_counter() - t)), len(out.details["tubes"]["kept"])


res = {}
for n_boxes in (1, 3):
    cold4, cold10, grow = [], [], []
    for _ in range(3):
        m = OnnxTemporalModel.from_package(pkg, embedding_cache_size=4096)
        cold4.append(job(m, 4, n_boxes))  # first submission of an event
        grow.append(
            [job(m, n, n_boxes)[0] for n in range(5, 11)]
        )  # same window, one frame longer each round
        m = OnnxTemporalModel.from_package(pkg, embedding_cache_size=4096)
        cold10.append(job(m, 10, n_boxes))
    res[f"{n_boxes}_tube"] = {
        "first_job_4_frames_ms": statistics.median(c[0] for c in cold4),
        "tubes": cold4[0][1],
        "next_jobs_ms_5_to_10_frames": [
            statistics.median(g[i] for g in grow) for i in range(6)
        ],
        "cold_10_frames_ms": statistics.median(c[0] for c in cold10),
    }
print(json.dumps(res, indent=1))

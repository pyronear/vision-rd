"""Time global pass vs global + 5 native 1024 tiles on a 4K frame (NCNN, 4 threads)."""

import json
import statistics
import subprocess
import sys
import time

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO

torch.set_num_threads(4)
N = 1024
src = cv2.imread("frame.jpg")
cv2.imwrite(
    "frame_4k.jpg", cv2.resize(src, (3840, 2160), interpolation=cv2.INTER_CUBIC)
)
m = YOLO(sys.argv[1], task="detect")
xs = [0, 704, 1408, 2112, 2816]
y0 = 568


def det(img):
    r = m.predict(img, imgsz=N, conf=0.05, verbose=False)[0]
    return torch.cat([r.boxes.xyxy, r.boxes.conf[:, None]], 1).cpu()


def once():
    t = {}
    a = time.perf_counter()
    im = cv2.imread("frame_4k.jpg")
    t["decode_jpeg"] = time.perf_counter() - a
    a = time.perf_counter()
    g = det(im)
    t["global"] = time.perf_counter() - a
    a = time.perf_counter()
    parts = [g] + [
        det(im[y0 : y0 + N, x : x + N]) + torch.tensor([x, y0, x, y0, 0.0]) for x in xs
    ]
    t["tiles_x5"] = time.perf_counter() - a
    a = time.perf_counter()
    d = torch.cat(parts)
    nms(d[:, :4], d[:, 4], 0.5)
    t["fusion"] = time.perf_counter() - a
    return t


for _ in range(3):
    once()
runs = [once() for _ in range(20)]
med = {k: round(1000 * statistics.median(r[k] for r in runs), 1) for k in runs[0]}
med["today_total"] = round(med["decode_jpeg"] + med["global"], 1)
med["multiscale_total"] = round(
    sum(med[k] for k in ("decode_jpeg", "global", "tiles_x5", "fusion")), 1
)
med["temp"] = subprocess.run(
    ["vcgencmd", "measure_temp"], capture_output=True, text=True
).stdout.strip()
med["throttled"] = subprocess.run(
    ["vcgencmd", "get_throttled"], capture_output=True, text=True
).stdout.strip()
print(json.dumps(med))

"""Endurance: every PERIOD s, 8 real 4K frames, engine path + 5 native tiles."""

import glob
import json
import subprocess
import sys
import time

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO

torch.set_num_threads(4)
cycles, period, N = int(sys.argv[1]), float(sys.argv[2]), 1024
paths = sorted(glob.glob("frames4k/*.jpg"))
m = YOLO("nimble_narwhal_ncnn_model", task="detect")
xs, y0 = [0, 704, 1408, 2112, 2816], 568


def det(img):
    r = m.predict(img, imgsz=N, conf=0.05, verbose=False)[0]
    return torch.cat([r.boxes.xyxy, r.boxes.conf[:, None]], 1).cpu()


def vc(*a):
    return (
        subprocess.run(["vcgencmd", *a], capture_output=True, text=True)
        .stdout.strip()
        .split("=")[-1]
    )


det(cv2.imread(paths[0]))  # warm-up
t0 = time.time()
for c in range(cycles):
    start = time.time()
    per_img = []
    n_det = 0
    for k in range(8):
        a = time.perf_counter()
        im = cv2.imread(paths[(8 * c + k) % len(paths)])
        small = cv2.resize(
            im, (1280, 720), interpolation=cv2.INTER_LINEAR
        )  # engine: frame_size
        cv2.imencode(
            ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )  # engine: upload JPEG
        parts = [det(small) * torch.tensor([3.0, 3.0, 3.0, 3.0, 1.0])]
        parts += [
            det(im[y0 : y0 + N, x : x + N]) + torch.tensor([x, y0, x, y0, 0.0])
            for x in xs
        ]
        d = torch.cat(parts)
        n_det += len(nms(d[:, :4], d[:, 4], 0.5))
        per_img.append(time.perf_counter() - a)
    busy = time.time() - start
    print(
        json.dumps(
            {
                "cycle": c,
                "t_min": round((start - t0) / 60, 1),
                "busy_s": round(busy, 2),
                "max_img_s": round(max(per_img), 2),
                "dets": n_det,
                "temp": vc("measure_temp"),
                "throttled": vc("get_throttled"),
                "arm_mhz": int(vc("measure_clock", "arm")) // 10**6,
            }
        ),
        flush=True,
    )
    time.sleep(max(period - (time.time() - start), 0))

"""Estimate the noise floor and bias of the registration estimator.

A. Synthetic: real SDIS 77 frames shifted by a known sub-pixel amount (cubic
   interpolation + JPEG re-encode), then registered. Measures bias such as
   integer locking, without any lighting change (optimistic bound).
B. Neighbour pairs: pyro-sdis brison-200 frames registered against the previous
   frame when they are <= --max-gap-s apart. The camera cannot move in that gap,
   so the measured offset is the real-world noise floor (upper bound: includes
   wind / thermal micro-motion).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from camera_pose_alignment.registration import prepare, register

SHIFTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.5, 5.3]


def synthetic(data_dir: Path, crop: float, n: int, seed: int) -> pd.DataFrame:
    meta = pd.read_csv(data_dir / "sequences.csv")
    files = meta.image_file.sample(n, random_state=seed)
    rows = []
    for f in files:
        gray = cv2.imread(str(data_dir / f), 0)
        h, w = gray.shape
        ref_g = prepare(gray, crop)
        for s in SHIFTS:
            m = np.float32([[1, 0, s], [0, 1, s / 2]])
            sh = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REFLECT)  # fmt: skip
            ok, buf = cv2.imencode(".jpg", sh, [cv2.IMWRITE_JPEG_QUALITY, 90])
            r = register(ref_g, prepare(cv2.imdecode(buf, 0), crop))
            if r is None:
                continue
            rows.append(dict(file=f, tx=s, ty=s / 2, dx=r[0], dy=r[1], cc=r[2]))
    return pd.DataFrame(rows)


def neighbours(camera: str, crop: float, max_gap_s: float) -> pd.DataFrame:
    from datasets import load_dataset  # heavy, only needed here

    ds = load_dataset("pyronear/pyro-sdis", split="train")
    meta = ds.remove_columns(["image", "annotations"]).to_pandas()
    meta["t"] = pd.to_datetime(meta.date, format="%Y-%m-%dT%H-%M-%S")
    meta = meta[meta.camera == camera].sort_values("t")
    gap = meta.t.diff().dt.total_seconds()
    rows, prev_g = [], None
    for (idx, row), g_s in zip(meta.iterrows(), gap, strict=True):
        g = prepare(cv2.cvtColor(np.array(ds[int(idx)]["image"].convert("RGB")),
                                 cv2.COLOR_RGB2GRAY), crop)  # fmt: skip
        if prev_g is not None and g_s <= max_gap_s:
            r = register(prev_g, g)
            if r is not None:
                rows.append(dict(t=row.t, gap_s=g_s, dx=r[0], dy=r[1], cc=r[2]))
        prev_g = g
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--crop-top", type=float, default=0.45)
    p.add_argument("--n-synthetic", type=int, default=200)
    p.add_argument("--camera", default="brison-200")
    p.add_argument("--max-gap-s", type=float, default=60)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic(a.data_dir, a.crop_top, a.n_synthetic, a.seed).to_csv(
        a.output_dir / "calib_synthetic.csv", index=False
    )
    neighbours(a.camera, a.crop_top, a.max_gap_s).to_csv(
        a.output_dir / "calib_neighbours.csv", index=False
    )


if __name__ == "__main__":
    main()

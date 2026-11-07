"""Register every image of the SDIS 77 export against its pose reference.

For each (camera, pose) series sorted by time, writes one row per image with:
- dx, dy, cc: offset vs the pose reference (phase-correlation seed + ECC)
- dx0, dy0, cc0: same with identity init (the original method, for comparison)
- pdx, pdy, pcc: offset vs the previous image of the same pose (reference-free)
- sat, bright: mean HSV saturation / value (sat ~ 0 => IR night image)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from camera_pose_alignment.registration import prepare, register


def nan3(r):
    return r if r is not None else (np.nan, np.nan, np.nan)


def run_pose(args):
    root, crop, rows = args
    ref_g = prepare(cv2.imread(str(root / rows[0]["pose_ref_file"]), 0), crop)
    out, prev_g = [], None
    for row in rows:
        bgr = cv2.imread(str(root / row["image_file"]))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        g = prepare(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), crop)
        dx, dy, cc = nan3(register(ref_g, g))
        dx0, dy0, cc0 = nan3(register(ref_g, g, coarse=False))
        pdx, pdy, pcc = nan3(register(prev_g, g) if prev_g is not None else None)
        prev_g = g
        out.append(
            dict(
                sequence_id=row["sequence_id"],
                dx=dx, dy=dy, cc=cc, dx0=dx0, dy0=dy0, cc0=cc0,
                pdx=pdx, pdy=pdy, pcc=pcc,
                sat=float(hsv[..., 1].mean()), bright=float(hsv[..., 2].mean()),
            )
        )  # fmt: skip
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crop-top", type=float, default=0.45)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    meta = pd.read_csv(a.data_dir / "sequences.csv").dropna(subset=["pose_ref_file"])
    meta["t"] = pd.to_datetime(meta.recorded_at, format="ISO8601")
    meta = meta.sort_values("t")
    jobs = [
        (a.data_dir, a.crop_top, g.to_dict("records"))
        for _, g in meta.groupby(["camera_id", "detection_pose_id"])
    ]
    with ProcessPoolExecutor(a.workers) as ex:
        res = [r for rows in ex.map(run_pose, jobs) for r in rows]
    df = meta.merge(pd.DataFrame(res), on="sequence_id")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"{len(df)} images -> {a.output}")


if __name__ == "__main__":
    main()

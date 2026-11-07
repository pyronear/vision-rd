"""Measure skyline tilt error with known angles and controlled defects (#160)."""

import argparse
import csv
import hashlib
import itertools
import json
from importlib.metadata import distribution, version
from pathlib import Path

import cv2
import numpy as np
from smoke_localisation import calibrate, load_dsm
from smoke_localisation.calibration import dem_skyline
from smoke_localisation.camera import pixel_angles
from smoke_localisation.dem import dsm_interpolator

POSES = (2, 13, 15)
TILT, ROLL, EXTENT = 2.0, 1.0, 50_000


def render(skyline, azimuths, shape, hfov, tilt, roll):
    """Render pixel centres using pixel_to_ray's rotation, independently of the fit."""
    h, w = shape
    x, y = pixel_angles(
        (np.arange(w)[None, :] + 0.5) / w,
        (np.arange(h)[:, None] + 0.5) / h,
        hfov,
        (w, h),
    )
    r = np.radians(roll)
    bearing = x * np.cos(r) + y * np.sin(r)
    elevation = y * np.cos(r) - x * np.sin(r) - tilt
    return (elevation > np.interp(bearing, azimuths, skyline)).astype(np.float32)


def shifted(mask, pixels):
    rows = np.clip(np.arange(mask.shape[0]) - pixels, 0, mask.shape[0] - 1)
    return mask[rows]


def cases():
    yield "perfect", {}
    yield "azimuth_on_grid_control", dict(on_grid=True)
    yield "zero_roll_control", {"roll": 0}
    for noise in (2, 4):
        for seed in range(3):
            yield f"dsm_noise_{noise}m", dict(noise=noise, seed=seed)
    yield "mask_384px", dict(small=True)
    for sign in (-1, 1):
        yield "edge_1px", dict(edge=sign)
        yield "edge_384px_1px", dict(small=True, edge=sign)
        yield "fov_1deg", dict(fov=sign)
        yield "canopy_5m", dict(canopy=sign * 5)
    for k in (0.10, 0.16):
        yield "refraction", dict(k=k)
    yield "cutoff_25km", dict(extent=25_000)
    for sign, edge, k, seed in itertools.product(
        (-1, 1), (-1, 1), (0.10, 0.16), range(3)
    ):
        yield (
            "combined",
            dict(
                small=True,
                edge=edge,
                fov=sign,
                k=k,
                seed=seed,
                noise=4,
                canopy=5,
                extent=25_000,
            ),
        )


def evaluate(cam, dsm, shape):
    lat, lon = float(cam["lat"]), float(cam["lon"])
    height = float(cam["mast_height_m"] or 20)
    hfov = float(cam["angle_of_view_deg_platform"])
    # Off-grid truth exposes azimuth discretisation; the supplied guess is 2° off.
    azimuth = float(cam["pose_azimuth_deg"]) + 0.037
    interp = dsm_interpolator(*dsm)
    camera_z = float(interp((lat, lon))) + height
    angles = np.arange(-hfov / 2 - 2, hfov / 2 + 2.025, 0.025)
    skyline = dem_skyline(
        lat, lon, camera_z, azimuth + angles, interp, max_distance_m=EXTENT
    )
    if not np.isfinite(skyline).all():
        raise ValueError("DSM does not cover the rendered skyline")
    masks = {
        roll: render(skyline, angles, shape, hfov, TILT, roll) for roll in (ROLL, 0)
    }
    # A square network input preserves the original camera's vertical angular FOV.
    small = cv2.resize(masks[ROLL], (384, 384), interpolation=cv2.INTER_AREA)
    for name, defects in cases():
        surface = dsm[0]
        if defects.get("noise"):
            noise = (
                np.random.default_rng(defects["seed"])
                .normal(0, defects["noise"], surface.shape)
                .astype(np.float32)
            )
            surface = surface + noise
        if defects.get("canopy"):
            surface = surface + defects["canopy"]
        model = (surface, *dsm[1:])
        # DSM defects must not also move the camera's absolute altitude.
        model_height = camera_z - float(dsm_interpolator(*model)((lat, lon)))
        mask = small if defects.get("small") else masks[defects.get("roll", ROLL)]
        mask = shifted(mask, defects.get("edge", 0))
        if defects.get("small"):
            mask = np.clip(
                cv2.resize(mask, shape[::-1], interpolation=cv2.INTER_LINEAR), 0, 1
            )
        row = dict(scenario=name, **defects)
        try:
            result = calibrate(
                mask,
                lat,
                lon,
                model_height,
                azimuth + 2
                if defects.get("on_grid")
                else float(cam["pose_azimuth_deg"]) + 2,
                hfov + defects.get("fov", 0),
                *model,
                max_skyline_distance_m=defects.get("extent", EXTENT),
                refraction_coefficient=defects.get("k", 0.13),
            )
            row.update(
                status="accepted",
                tilt_error_deg=result.tilt_deg - TILT,
                azimuth_error_deg=(result.azimuth_deg - azimuth + 180) % 360 - 180,
                residual_deg=result.residual_deg,
            )
        except ValueError as exc:
            row.update(status="rejected", reason=str(exc))
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/02_intermediate/dem"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/08_reporting/skyline-floor")
    )
    args = parser.parse_args()
    cameras = list(csv.DictReader((args.input / "cameras.csv").open()))
    images = list(csv.DictReader((args.input / "images.csv").open()))
    if len(cameras) < max(POSES):
        parser.error("Expected the #159 bundle's camera row order")
    args.output.mkdir(parents=True, exist_ok=True)
    records, provenance = (
        [],
        dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runtime=json.loads(
                distribution("smoke-localisation").read_text("direct_url.json")
            ),
            versions={
                p: version(p)
                for p in (
                    "numpy",
                    "scipy",
                    "opencv-python-headless",
                    "pyproj",
                    "rasterio",
                )
            },
            manifests={
                p: hashlib.sha256((args.input / p).read_bytes()).hexdigest()
                for p in ("cameras.csv", "images.csv")
            },
            truth=dict(tilt_deg=TILT, roll_deg=ROLL, refraction=0.13, extent_m=EXTENT),
            poses={},
        ),
    )
    with (args.output / "trials.jsonl").open("w", buffering=1) as stream:
        for index in POSES:
            pose, cam = f"P{index:02d}", cameras[index - 1]
            image = next(r for r in images if r["pose_id"] == cam["pose_id"])
            path = (args.input / image["file"]).resolve()
            if not path.is_relative_to(args.input.resolve()):
                raise ValueError("Image path escapes the supplied bundle")
            decoded = cv2.imread(str(path))
            if decoded is None:
                raise ValueError("Image cannot be decoded")
            shape = decoded.shape[:2]
            dsm = load_dsm(float(cam["lat"]), float(cam["lon"]), 0.7, args.cache)
            provenance["poses"][pose] = dict(
                camera=cam,
                shape=shape,
                dsm_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in dsm],
            )
            for row in evaluate(cam, dsm, shape):
                row = dict(pose=pose, **row)
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(
                    pose,
                    row["scenario"],
                    row.get("tilt_error_deg", row["status"]),
                    flush=True,
                )
    summary = []
    for pose, name in dict.fromkeys((r["pose"], r["scenario"]) for r in records):
        trials = [r for r in records if r["pose"] == pose and r["scenario"] == name]
        errors = [r["tilt_error_deg"] for r in trials if r["status"] == "accepted"]
        summary.append(
            dict(
                pose=pose,
                scenario=name,
                trials=len(trials),
                accepted=len(errors),
                max_abs_tilt_error_deg=max(map(abs, errors)) if errors else None,
                min_tilt_error_deg=min(errors) if errors else None,
                max_tilt_error_deg=max(errors) if errors else None,
            )
        )
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

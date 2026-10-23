"""Compare calibration sensitivity on real mast terrain and paired flat controls."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
from pyproj import Geod
from smoke_localisation.dem import dsm_interpolator, load_dsm
from smoke_localisation.ray import curvature_drop_m, ray_ground_intersection

from study import pose, ray

GEOD = Geod(ellps="WGS84")
LEVELS = {
    "azimuth": (0.001, 0.003, 0.01, 0.03, 0.1),
    "tilt": (0.001, 0.003, 0.01, 0.03, 0.1),
    "roll": (0.001, 0.003, 0.01, 0.03, 0.1),
    "hfov": (0.001, 0.003, 0.01, 0.03, 0.1),
    "height": (0.1, 0.3, 1, 3, 10),
    "position_east": (0.1, 0.3, 1, 3, 10),
    "position_north": (0.1, 0.3, 1, 3, 10),
}
PIXELS = ((0.5, 0.5), (0.1, 0.1), (0.9, 0.9))


def sweep_view(site, dsm, bearing, distance, budget):
    """Aim at a known surface point; retain blocked views and missing projections."""
    lat, lon, height = (site[k] for k in ("latitude", "longitude", "mast_height_m"))
    target_lon, target_lat, _ = GEOD.fwd(lon, lat, bearing, distance)
    interp = dsm_interpolator(*dsm)
    camera_z, target_z = (
        float(interp((lat, lon))),
        float(interp((target_lat, target_lon))),
    )
    if not all(math.isfinite(z) for z in (camera_z, target_z)):
        raise ValueError("DSM does not cover the camera and target")
    elevation = math.degrees(
        math.atan2(target_z - camera_z - height - curvature_drop_m(distance), distance)
    )

    def cast(azimuth, tilt, mast):
        return ray_ground_intersection(
            lat,
            lon,
            mast,
            azimuth,
            elevation - tilt,
            *dsm,
            max_distance_m=15_000,
            step_m=10,
        )

    view = dict(
        site=site["site"],
        support_id=site["support_id"],
        bearing_deg=bearing,
        distance_m=distance,
        mast_height_m=height,
        target_lat=target_lat,
        target_lon=target_lon,
        target_elevation_m=target_z,
    )
    baseline = cast(bearing, 0, height)
    offset = (
        GEOD.inv(target_lon, target_lat, baseline[1], baseline[0])[2]
        if baseline
        else None
    )
    view.update(nominal_offset_m=offset, status="visible")
    # shortcut: one DSM-cell tolerance; confirm visibility on site before acceptance.
    if offset is None or offset > 30:
        view["status"] = "occluded_or_unresolved"
        return view, [], []
    errors, precision = [], []
    for index, (parameter, levels) in enumerate(LEVELS.items()):
        passing = []
        for amplitude in levels:
            pair = []
            for pixel, sign in itertools.product(PIXELS, (-1, 1)):
                delta = sign * amplitude
                azimuth, tilt = pose(*pixel, 54.2, math.radians(elevation))
                angles = [azimuth + bearing, tilt, 0, 54.2]
                cam_lat, cam_lon, mast = lat, lon, height
                if index < 4:
                    angles[index] += delta
                elif parameter == "height":
                    mast += delta
                else:
                    cam_lon, cam_lat, _ = GEOD.fwd(
                        lon, lat, 90 if parameter == "position_east" else 0, delta
                    )
                    # Keep absolute camera altitude fixed for horizontal GPS errors.
                    mast += camera_z - float(interp((cam_lat, cam_lon)))
                dx, dy, dz = ray(*pixel, *angles)
                hit = (
                    ray_ground_intersection(
                        cam_lat,
                        cam_lon,
                        mast,
                        math.degrees(math.atan2(dx, dy)),
                        math.degrees(math.atan2(dz, math.hypot(dx, dy))),
                        *dsm,
                        max_distance_m=15_000,
                        step_m=10,
                    )
                    if mast > 0
                    else None
                )
                error = (
                    GEOD.inv(baseline[1], baseline[0], hit[1], hit[0])[2]
                    if hit
                    else None
                )
                pair.append(error)
                errors.append(
                    view
                    | dict(
                        parameter=parameter,
                        delta=delta,
                        u=pixel[0],
                        v=pixel[1],
                        unit="deg" if index < 4 else "m",
                        status="hit" if hit else "no_hit",
                        error_m=error,
                    )
                )
            passing.append(all(e is not None and e <= budget for e in pair))
        prefix = len(list(itertools.takewhile(bool, passing)))
        precision.append(
            view
            | dict(
                parameter=parameter,
                unit="deg" if index < 4 else "m",
                budget_m=budget,
                tested_limit=levels[prefix - 1] if prefix else 0,
                limit_status="all_tested_pass"
                if prefix == len(levels)
                else "below_grid"
                if not prefix
                else "sampled_only",
            )
        )
    return view, errors, precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites", type=Path, default=Path(__file__).with_name("sites.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/08_reporting/terrain")
    )
    parser.add_argument("--cache", type=Path, default=Path("data/01_raw/dem"))
    parser.add_argument("--budget", type=float, default=100)
    args = parser.parse_args()
    if not math.isfinite(args.budget) or args.budget <= 0:
        parser.error("Budget must be finite and positive")
    with args.sites.open() as stream:
        sites = list(csv.DictReader(stream))
    if not sites:
        parser.error("Supply at least one mast")
    for site in sites:
        for key, low, high in (
            ("latitude", -85, 85),
            ("longitude", -180, 180),
            ("mast_height_m", 10.01, 1000),
        ):
            site[key] = float(site[key])
            if not math.isfinite(site[key]) or not low <= site[key] <= high:
                parser.error(f"{key} must be finite and in [{low}, {high}]")
    views, errors, precision, summaries = [], [], [], []
    for site in sites:
        dsm = load_dsm(site["latitude"], site["longitude"], 0.25, args.cache)
        if not np.isfinite(dsm[0]).all():
            parser.error(f"{site['site']}: DSM has missing cells")
        ground = float(dsm_interpolator(*dsm)((site["latitude"], site["longitude"])))
        for model in ("flat_control", "DSM"):
            terrain = (
                dsm if model == "DSM" else (np.full_like(dsm[0], ground), *dsm[1:])
            )
            for distance in (2000, 5000, 10_000):
                group_views, group_errors, group_precision = [], [], []
                for bearing in range(0, 360, 45):
                    view, cases, limits = sweep_view(
                        site, terrain, bearing, distance, args.budget
                    )
                    group_views.append(view | {"model": model})
                    group_errors.extend(c | {"model": model} for c in cases)
                    group_precision.extend(c | {"model": model} for c in limits)
                tilt = [
                    c
                    for c in group_errors
                    if c["parameter"] == "tilt" and abs(c["delta"]) == 0.1
                ]
                hits = [c["error_m"] for c in tilt if c["error_m"] is not None]
                tilt_precision = [
                    c for c in group_precision if c["parameter"] == "tilt"
                ]
                summaries.append(
                    dict(
                        site=site["site"],
                        model=model,
                        distance_m=distance,
                        targets=len(group_views),
                        visible=sum(v["status"] == "visible" for v in group_views),
                        tilt_0p1_max_error_m=max(hits) if hits else None,
                        tilt_0p1_missing_hits=sum(c["error_m"] is None for c in tilt),
                        tilt_all_tested_within_budget=sum(
                            c["limit_status"] == "all_tested_pass"
                            for c in tilt_precision
                        ),
                        tilt_grid_prefix_deg=min(
                            c["tested_limit"] for c in tilt_precision
                        )
                        if tilt_precision
                        else None,
                    )
                )
                views.extend(group_views)
                errors.extend(group_errors)
                precision.extend(group_precision)
                print(
                    f"{site['site']} {model} {distance} m: "
                    f"{summaries[-1]['visible']}/8 targets visible",
                    flush=True,
                )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("views", views),
        ("errors", errors),
        ("precision", precision),
        ("summary", summaries),
    ):
        with (args.output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    metadata = dict(
        python=sys.version,
        budget_m=args.budget,
        step_m=10,
        max_distance_m=15_000,
        nominal_tolerance_m=30,
        levels=LEVELS,
        hfov_deg=54.2,
        aspect_ratio=[16, 9],
        pixels=PIXELS,
        prototype=json.loads(
            distribution("smoke-localisation").read_text("direct_url.json")
        ),
        versions={
            p: version(p)
            for p in ("smoke-localisation", "numpy", "pyproj", "rasterio", "scipy")
        },
        sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), args.sites, *sorted(args.cache.glob("*.tif")))
        },
    )
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Measure tilt repeatability, rejections and candidate camera moves."""

import argparse
import csv
import hashlib
import json
import math
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import smoke_localisation
import yaml
from pyproj import Geod
from requests import HTTPError
from smoke_localisation import calibrate, detect_sky, load_dsm
from smoke_localisation.dem import dsm_interpolator
from smoke_localisation.ray import curvature_drop_m, ray_ground_intersection

GEOD = Geod(ellps="WGS84")


def sigma(values):
    a = np.asarray(values, dtype=float)
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else None


def angular_distance(a, b):
    return abs((a - b + 180) % 360 - 180)


def circular_median(values):
    return min(values, key=lambda x: sum(angular_distance(x, y) for y in values))


def tilt_range(profile, distance, cfg):
    key = f"required_tilt_{distance}m_deg"
    if key not in profile:
        return "no_reference_fit", None, None
    last_pass = profile[key]
    if last_pass is None:
        return "target_not_reached", None, None
    grid = cfg["tilt_grid_deg"]
    if last_pass == 0:
        return "below_grid", None, grid[0]
    first_fail = next((x for x in grid if x > last_pass), None)
    if first_fail is None:
        return "cap_reached", last_pass, None
    return "bracketed", last_pass, first_fail


def epochs(frames, cfg):
    """Flag persistent level changes as candidate camera moves."""
    n = cfg["change_min_frames"]
    cuts = [0, len(frames)]
    pending = [(0, len(frames))]
    while pending:
        start, end = pending.pop()
        candidates = []
        for cut in range(start + n, end - n + 1):
            left = [r["tilt_deg"] for r in frames[cut - n : cut]]
            right = [r["tilt_deg"] for r in frames[cut : cut + n]]
            jump = abs(float(np.median(left) - np.median(right)))
            noise = max(
                sigma(left),
                sigma(right),
                float(
                    np.median(
                        [r["tilt_uncertainty_deg"] for r in frames[cut - n : cut + n]]
                    )
                ),
            )
            if jump > max(
                cfg["epoch_min_shift_deg"], cfg["epoch_noise_multiple"] * noise
            ):
                before = np.array([r["tilt_deg"] for r in frames[start:cut]])
                after = np.array([r["tilt_deg"] for r in frames[cut:end]])
                cost = (
                    np.abs(before - np.median(before)).sum()
                    + np.abs(after - np.median(after)).sum()
                )
                candidates.append((-float(cost), cut))
        if candidates:
            _, cut = max(candidates)
            cuts.append(cut)
            pending.extend(((start, cut), (cut, end)))
    ordered = sorted(cuts)
    groups = [frames[a:b] for a, b in pairwise(ordered) if b > a]
    i = 0
    while i < len(groups) - 1:
        shift = abs(
            float(
                np.median([r["tilt_deg"] for r in groups[i]])
                - np.median([r["tilt_deg"] for r in groups[i + 1]])
            )
        )
        if shift < cfg["epoch_min_shift_deg"]:
            groups[i].extend(groups.pop(i + 1))
            i = max(0, i - 1)
        else:
            i += 1
    return groups


def fit(mask, cam, height, hfov, dsm, cfg, **overrides):
    kwargs = dict(
        max_azimuth_offset_deg=cfg["azimuth_search_deg"],
        max_residual_deg=cfg["max_residual_deg"],
        # Keep this study's baseline comparable with the original 25 km run.
        max_skyline_distance_m=25000,
    )
    kwargs.update(overrides)
    try:
        return calibrate(
            mask,
            float(cam["lat"]),
            float(cam["lon"]),
            height,
            float(cam["pose_azimuth_deg"]),
            hfov,
            *dsm,
            **kwargs,
        )
    except ValueError as exc:
        if "search boundary" not in str(exc):
            raise
        kwargs["max_azimuth_offset_deg"] = cfg["azimuth_retry_deg"]
        return calibrate(
            mask,
            float(cam["lat"]),
            float(cam["lon"]),
            height,
            float(cam["pose_azimuth_deg"]),
            hfov,
            *dsm,
            **kwargs,
        )


def tilt_requirement(cam, height, bearing, dsm, distance, cfg):
    lat, lon = float(cam["lat"]), float(cam["lon"])
    target_lon, target_lat, _ = GEOD.fwd(lon, lat, bearing, distance)
    interp = dsm_interpolator(*dsm)
    elevation = math.degrees(
        math.atan2(
            float(interp((target_lat, target_lon)))
            - float(interp((lat, lon)))
            - height
            - curvature_drop_m(distance),
            distance,
        )
    )

    def cast(delta):
        return ray_ground_intersection(
            lat,
            lon,
            height,
            bearing,
            elevation - delta,
            *dsm,
            max_distance_m=15000,
            step_m=10,
        )

    nominal = cast(0)
    if (
        nominal is None
        or GEOD.inv(target_lon, target_lat, nominal[1], nominal[0])[2] > 30
    ):
        return None
    passed = 0.0
    for amplitude in cfg["tilt_grid_deg"]:
        hits = [cast(sign * amplitude) for sign in (-1, 1)]
        if not all(
            h is not None
            and GEOD.inv(nominal[1], nominal[0], h[1], h[0])[2]
            <= cfg["position_budget_m"]
            for h in hits
        ):
            break
        passed = amplitude
    return passed


def run(args, cfg):
    cameras = list(csv.DictReader((args.input / "cameras.csv").open()))
    images = list(csv.DictReader((args.input / "images.csv").open()))
    poses = {c["pose_id"] for c in cameras}
    if len(poses) != len(cameras) or poses != {r["pose_id"] for r in images}:
        raise ValueError(
            "Camera and image manifests must contain the same unique poses"
        )
    if any(not pose or Path(pose).name != pose for pose in poses):
        raise ValueError("Pose IDs must be safe filename components")
    if len({r["file"] for r in images}) != len(images):
        raise ValueError("Duplicate image paths in the input manifest")
    for row in images:
        path = (args.input / row["file"]).resolve()
        if not path.is_relative_to(args.input.resolve()) or not path.is_file():
            raise ValueError("Image path escapes the bundle or does not exist")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.cache / "masks").mkdir(parents=True, exist_ok=True)
    package = Path(smoke_localisation.__file__).parent
    sources = [
        Path(__file__),
        args.params,
        args.input / "cameras.csv",
        args.input / "images.csv",
        *package.glob("*.py"),
        *(package / "models" / "skyseg").glob("*"),
    ]
    provenance = dict(
        config=cfg,
        sources={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        versions={
            name: version(name)
            for name in (
                "smoke-localisation",
                "numpy",
                "scipy",
                "ncnn",
                "opencv-python-headless",
                "rasterio",
                "pyproj",
            )
        },
        images={
            r["file"]: hashlib.sha256((args.input / r["file"]).read_bytes()).hexdigest()
            for r in images
        },
    )
    saved = args.output / "provenance.json"
    if saved.exists() and json.loads(saved.read_text()) != provenance:
        raise ValueError("Run provenance changed; use a new output directory")
    saved.write_text(json.dumps(provenance, indent=2) + "\n")
    output = args.output / "frames.jsonl"
    records = (
        {r["file"]: r for r in map(json.loads, output.read_text().splitlines())}
        if output.exists()
        else {}
    )
    profile_path = args.output / "profiles.json"
    profiles = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    camera_codes = {
        name: f"C{i + 1:02d}"
        for i, name in enumerate(dict.fromkeys(c["platform_camera"] for c in cameras))
    }
    terrains, previous_camera = {}, None
    for i, cam in enumerate(cameras):
        pose = cam["pose_id"]
        if cam["platform_camera"] != previous_camera:
            terrains.clear()
            previous_camera = cam["platform_camera"]

        def terrain(margin, cam=cam):
            if margin not in terrains:
                terrains[margin] = load_dsm(
                    float(cam["lat"]), float(cam["lon"]), margin, args.cache / "dem"
                )
            return terrains[margin]

        def mask_for(row, pose=pose):
            path = args.cache / "masks" / (pose + "_" + Path(row["file"]).stem + ".npz")
            if path.exists():
                return np.load(path)["mask"]
            image = cv2.imread(str(args.input / row["file"]))
            if image is None:
                raise ValueError("Image cannot be decoded")
            mask = detect_sky(image)
            np.savez_compressed(path, mask=mask)
            return mask

        dsm = terrain(cfg["dem_margin_deg"])
        height = (
            float(cam["mast_height_m"])
            if cam["mast_height_m"]
            else cfg["assumed_height_m"]
        )
        hfov = float(cam["angle_of_view_deg_platform"])
        rows = sorted(
            [r for r in images if r["pose_id"] == pose], key=lambda r: r["datetime_utc"]
        )
        with output.open("a", buffering=1) as stream:
            for row in rows:
                if row["file"] in records:
                    continue
                record = dict(
                    row,
                    pose=f"P{i + 1:02d}",
                    camera=camera_codes[cam["platform_camera"]],
                    height_known=bool(cam["mast_height_m"]),
                    height_m=height,
                    hfov_deg=hfov,
                )
                try:
                    c = fit(mask_for(row), cam, height, hfov, dsm, cfg)
                    record.update(
                        status="accepted",
                        tilt_deg=c.tilt_deg,
                        roll_deg=c.roll_deg,
                        azimuth_deg=c.azimuth_deg,
                        residual_deg=c.residual_deg,
                        tilt_uncertainty_deg=c.tilt_uncertainty_deg,
                        azimuth_uncertainty_deg=c.azimuth_uncertainty_deg,
                    )
                except ValueError as exc:
                    record.update(status="rejected", reason=str(exc))
                records[row["file"]] = record
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        if pose not in profiles:
            reference = next(
                (
                    records[r["file"]]
                    for r in rows
                    if records[r["file"]]["status"] == "accepted"
                ),
                None,
            )
            profile = dict(hfov=hfov, fov_profile_complete=False)
            if reference:
                mask = mask_for(reference)
                scenarios = dict(
                    fov=[dict(hfov_deg=hfov + d) for d in cfg["fov_offsets_deg"] if d],
                    height=[
                        dict(height_m=h)
                        for h in (
                            [
                                height - cfg["known_height_delta_m"],
                                height + cfg["known_height_delta_m"],
                            ]
                            if cam["mast_height_m"]
                            else cfg["unknown_height_scenarios_m"]
                        )
                    ],
                    mask=[dict(threshold=t) for t in cfg["mask_threshold_scenarios"]],
                    refraction=[
                        dict(refraction_coefficient=k)
                        for k in cfg["refraction_scenarios"]
                    ],
                    extent=[
                        dict(max_skyline_distance_m=cfg["wide_skyline_distance_m"])
                    ],
                )
                profile["reference_file"] = reference["file"]
                for name, variants in scenarios.items():
                    shifts, failures = [], []
                    for variant in variants:
                        h = variant.pop("height_m", height)
                        f = variant.pop("hfov_deg", hfov)
                        try:
                            surface = (
                                terrain(cfg["wide_dem_margin_deg"])
                                if name == "extent"
                                else dsm
                            )
                            c = fit(mask, cam, h, f, surface, cfg, **variant)
                            shifts.append(abs(c.tilt_deg - reference["tilt_deg"]))
                        except (ValueError, HTTPError) as exc:
                            failures.append(str(exc))
                    profile[name + "_sensitivity_deg"] = (
                        max(shifts) if shifts and not failures else None
                    )
                    profile[name + "_failures"] = failures
                profile["fov_profile_complete"] = not profile["fov_failures"]
                for distance in cfg["range_targets_m"]:
                    profile[f"required_tilt_{distance}m_deg"] = tilt_requirement(
                        cam, height, reference["azimuth_deg"], dsm, distance, cfg
                    )
            profiles[pose] = profile
            profile_path.write_text(
                json.dumps(profiles, indent=2, allow_nan=False) + "\n"
            )
        count = sum(records[r["file"]]["status"] == "accepted" for r in rows)
        print(f"P{i + 1:02d}: {count}/{len(rows)} fits", flush=True)


def summarize(args, cfg):
    frames = list(
        map(json.loads, (args.output / "frames.jsonl").read_text().splitlines())
    )
    profiles = json.loads((args.output / "profiles.json").read_text())
    summary, segments = [], []
    for pose in sorted({r["pose"] for r in frames}):
        all_rows = [r for r in frames if r["pose"] == pose]
        accepted = sorted(
            [r for r in all_rows if r["status"] == "accepted"],
            key=lambda r: r["datetime_utc"],
        )
        groups = epochs(accepted, cfg) if accepted else []
        profile = profiles[all_rows[0]["pose_id"]]
        pose_tilt = (
            float(np.median([r["tilt_deg"] for r in accepted])) if accepted else None
        )
        tilt_outliers = [r for r in accepted if abs(r["tilt_deg"] - pose_tilt) > 1]
        pose_azimuth = (
            circular_median([r["azimuth_deg"] for r in accepted]) if accepted else None
        )
        azimuth_outliers = sum(
            angular_distance(r["azimuth_deg"], pose_azimuth) > 0.5
            for r in tilt_outliers
        )
        uncertainty_coverage = (
            sum(
                abs(r["tilt_deg"] - pose_tilt) <= r["tilt_uncertainty_deg"]
                for r in accepted
            )
            / len(accepted)
            if accepted
            else None
        )
        requirements = {
            distance: tilt_range(profile, distance, cfg)
            for distance in cfg["range_targets_m"]
        }
        sigmas, tails = [], []
        for k, group in enumerate(groups):
            spread = sigma([r["tilt_deg"] for r in group])
            if len(group) >= cfg["epoch_min_frames"]:
                sigmas.append(spread)
                deviations = [
                    abs(
                        r["tilt_deg"]
                        - float(np.median([q["tilt_deg"] for q in group if q is not r]))
                    )
                    for r in group
                ]
                tails.append(float(np.quantile(deviations, 0.95)))
            segments.append(
                dict(
                    pose=pose,
                    epoch=k + 1,
                    n=len(group),
                    first=group[0]["datetime_utc"],
                    last=group[-1]["datetime_utc"],
                    median_tilt_deg=float(np.median([r["tilt_deg"] for r in group])),
                    repeatability_sigma_deg=spread
                    if len(group) >= cfg["epoch_min_frames"]
                    else None,
                )
            )
        summary.append(
            dict(
                pose=pose,
                camera=all_rows[0]["camera"],
                frames=len(all_rows),
                accepted=len(accepted),
                candidate_epochs=len(groups),
                height_known=all_rows[0]["height_known"],
                hfov_deg=profile["hfov"],
                fov_profile_complete=profile["fov_profile_complete"],
                repeatability_sigma_deg=max(sigmas) if sigmas else None,
                repeatability_p95_deg=max(tails) if tails else None,
                tilt_gt1deg_frames=len(tilt_outliers) if accepted else None,
                tilt_gt1deg_azimuth_gt05deg_frames=azimuth_outliers
                if accepted
                else None,
                tilt_uncertainty_within_pose_median_fraction=uncertainty_coverage,
                median_conditional_radius_deg=float(
                    np.median([r["tilt_uncertainty_deg"] for r in accepted])
                )
                if accepted
                else None,
                absolute_bias_deg=None,
                **{
                    f"requirement_{distance}m_status": result[0]
                    for distance, result in requirements.items()
                },
                **{
                    f"tilt_{distance}m_{bound}_deg": value
                    for distance, (_, last, first) in requirements.items()
                    for bound, value in (("last_pass", last), ("first_fail", first))
                },
                **{
                    key: profile.get(key)
                    for key in (
                        "fov_sensitivity_deg",
                        "height_sensitivity_deg",
                        "mask_sensitivity_deg",
                        "refraction_sensitivity_deg",
                        "extent_sensitivity_deg",
                    )
                },
            )
        )
    for name, rows in (("summary", summary), ("epochs", segments)):
        if rows:
            with (args.output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    analysis = dict(
        config=cfg,
        study_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frames_sha256=hashlib.sha256(
            (args.output / "frames.jsonl").read_bytes()
        ).hexdigest(),
    )
    (args.output / "analysis-provenance.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    reasons = {}
    for row in frames:
        if row["status"] == "rejected":
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    print(
        json.dumps(
            dict(
                frames=len(frames),
                accepted=sum(r["accepted"] for r in summary),
                poses=len(summary),
                rejections=reasons,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/08_reporting"))
    parser.add_argument("--cache", type=Path, default=Path("data/02_intermediate"))
    parser.add_argument(
        "--params", type=Path, default=Path(__file__).with_name("repeatability.yaml")
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.params.read_text())
    if cfg["change_min_frames"] < 2 or cfg["epoch_min_frames"] < 3:
        parser.error("Invalid change or repeatability sample size")
    if not args.summarize:
        run(args, cfg)
    summarize(args, cfg)


if __name__ == "__main__":
    main()

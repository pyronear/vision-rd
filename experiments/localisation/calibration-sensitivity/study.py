# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One-at-a-time calibration errors; local metres, pinhole camera, profile terrain."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from functools import partial
from pathlib import Path


def ray(u, v, azimuth, tilt, roll, hfov, vfov=None):
    """Right/forward/up ray; explicit vertical FOV or the study's 16:9 model."""
    a, t, r = map(math.radians, (azimuth, tilt, roll))
    scale = math.tan(math.radians(hfov) / 2)
    x, z = (2 * u - 1) * scale, (1 - 2 * v) * scale * 9 / 16
    if vfov is not None:
        z = (1 - 2 * v) * math.tan(math.radians(vfov) / 2)
    x, z = x * math.cos(r) + z * math.sin(r), z * math.cos(r) - x * math.sin(r)
    y, z = math.cos(t) + z * math.sin(t), z * math.cos(t) - math.sin(t)
    x, y = x * math.cos(a) + y * math.sin(a), y * math.cos(a) - x * math.sin(a)
    length = math.hypot(x, y, z)
    return x / length, y / length, z / length


def project(direction, azimuth, tilt, roll, hfov, vfov):
    """Inverse of ray(): place a known direction in the actual camera image."""
    x, y, z = direction
    a, t, r = map(math.radians, (azimuth, tilt, roll))
    x, y = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
    y, z = y * math.cos(t) - z * math.sin(t), y * math.sin(t) + z * math.cos(t)
    x, z = x * math.cos(r) - z * math.sin(r), x * math.sin(r) + z * math.cos(r)
    if y <= 0:
        return None
    u = 0.5 + x / (2 * y * math.tan(math.radians(hfov) / 2))
    v = 0.5 - z / (2 * y * math.tan(math.radians(vfov) / 2))
    return (u, v) if 0 <= u <= 1 and 0 <= v <= 1 else None


def pose(u, v, hfov, elevation):
    """Place the same known target at each image location, facing forward."""
    scale = math.tan(math.radians(hfov) / 2)
    x, z = (2 * u - 1) * scale, (1 - 2 * v) * scale * 9 / 16
    tilt = math.atan(z) - math.asin(
        math.hypot(x, 1, z) * math.sin(elevation) / math.hypot(1, z)
    )
    azimuth = -math.atan2(x, math.cos(tilt) + z * math.sin(tilt))
    return math.degrees(azimuth), math.degrees(tilt)


def elevation_at(profile, distance):
    for (y0, z0), (y1, z1) in itertools.pairwise(profile):
        if y0 <= distance <= y1:
            return z0 + (z1 - z0) * (distance - y0) / (y1 - y0)
    raise ValueError(f"Terrain does not cover {distance} m")


def intersect(direction, camera, profile, max_range):
    """First forward hit on piecewise linear terrain extruded sideways."""
    dx, dy, dz = direction
    cx, cy, cz = camera
    try:
        if cz <= elevation_at(profile, cy):
            return None
    except ValueError:
        return None
    hits = []
    for (y0, z0), (y1, z1) in itertools.pairwise(profile):
        slope = (z1 - z0) / (y1 - y0)
        denominator = dz - slope * dy
        if abs(denominator) < 1e-14:
            continue
        t = (z0 + slope * (cy - y0) - cz) / denominator
        x, y = cx + t * dx, cy + t * dy
        if (
            t > 0
            and y0 - 1e-8 <= y <= y1 + 1e-8
            and t * math.hypot(dx, dy) <= max_range
        ):
            hits.append((t, x, y))
    if not hits:
        return None
    _, x, y = min(hits)
    return x, y


def displacement(parameter, camera, angles, pixel, profile, distance, max_range, delta):
    perturbed_camera, perturbed_angles = list(camera), list(angles)
    if parameter < 4:
        perturbed_angles[parameter] += delta
    else:
        perturbed_camera[(2, 0, 1)[parameter - 4]] += delta
    hit = intersect(
        ray(*pixel, *perturbed_angles), perturbed_camera, profile, max_range
    )
    return None if hit is None else (hit[0], hit[1] - distance)


def tolerance(error, budget, cap):
    """First sampled budget crossing, then bisection, in both error directions."""
    limits = []
    for sign in (-1, 1):
        low, high = 0.0, min(1e-8, cap)
        while True:
            shift = error(sign * high)
            if shift is None or math.hypot(*shift) > budget:
                break
            low = high
            if high == cap:
                break
            high = min(high * 2, cap)
        if low < high:
            # ponytail: local monotonic branch; full uncertainty propagation is #113.
            for _ in range(35):
                middle = (low + high) / 2
                shift = error(sign * middle)
                if shift is None or math.hypot(*shift) > budget:
                    high = middle
                else:
                    low = middle
        limits.append(low)
    return min(limits), all(limit == cap for limit in limits)


def check():
    """Analytical cases, off-axis rotations, and a nearer ridge must agree."""
    flat = [(-30_000, 0), (30_000, 0)]
    for u, v, hfov in itertools.product((0.1, 0.5, 0.9), (0.1, 0.5, 0.9), (54.2, 87)):
        azimuth, tilt = pose(u, v, hfov, -math.atan2(35, 5000))
        hit = intersect(ray(u, v, azimuth, tilt, 0, hfov), (0, 0, 35), flat, 30_000)
        assert hit is not None and math.hypot(hit[0], hit[1] - 5000) < 1e-8
    tilt = math.degrees(math.atan2(35, 5000))
    angles, camera, pixel = (0, tilt, 0, 87), (0, 0, 35), (0.5, 0.5)
    for sign in (-1, 1):
        shift = displacement(1, camera, angles, pixel, flat, 5000, 30_000, sign * 0.01)
        expected = 35 / math.tan(math.radians(tilt + sign * 0.01)) - 5000
        assert shift is not None and math.isclose(shift[1], expected, abs_tol=1e-8)
    shift = displacement(4, camera, angles, pixel, flat, 5000, 30_000, 1)
    assert shift is not None and math.isclose(shift[1], 5000 / 35, abs_tol=1e-8)
    shift = displacement(0, camera, angles, pixel, flat, 5000, 30_000, 0.1)
    assert shift is not None and math.isclose(
        shift[0], 5000 * math.sin(math.radians(0.1))
    )
    for parameter, expected in ((5, (1, 0)), (6, (0, 1))):
        shift = displacement(parameter, camera, angles, pixel, flat, 5000, 30_000, 1)
        assert shift is not None and math.dist(shift, expected) < 1e-8
    for parameter, off_axis_pixel, sign in ((2, (0.75, 0.5), -1), (3, (0.5, 0.25), 1)):
        azimuth, off_axis_tilt = pose(*off_axis_pixel, 87, -math.atan2(35, 5000))
        shift = displacement(
            parameter,
            camera,
            (azimuth, off_axis_tilt, 0, 87),
            off_axis_pixel,
            flat,
            5000,
            30_000,
            0.01,
        )
        assert shift is not None and sign * shift[1] > 0
    assert intersect(ray(0.5, 0.5, 0, -1, 0, 87), camera, flat, 30_000) is None
    assert intersect(ray(0.5, 0.5, 0, tilt, 0, 87), camera, flat, 4000) is None
    assert all(
        math.isclose(a, b, abs_tol=1e-12)
        for a, b in zip(
            ray(0.5, 0.5, *angles), ray(0.5, 0.5, 0, tilt, 12, 54.2), strict=True
        )
    )
    assert math.isclose(ray(1, 0.5, 0, 0, 90, 90)[2], -math.sqrt(0.5))
    dx, dy, _ = ray(0.75, 0.5, 0, 0, 0, 87)
    assert math.isclose(dx / dy, 0.5 * math.tan(math.radians(87) / 2))
    uphill = [(-30_000, -3000), (30_000, 3000)]
    hit = intersect(ray(0.5, 0.5, *angles), camera, uphill, 30_000)
    assert hit is not None and math.isclose(hit[1], 35 / (0.1 + 35 / 5000))
    assert intersect(ray(0.5, 0.5, *angles), (0, 0, -1), flat, 30_000) is None
    ridge = [(0, 0), (1000, 30), (2000, 0), (30_000, 0)]
    hit = intersect(ray(0.5, 0.5, *angles), camera, ridge, 30_000)
    assert hit is not None and hit[1] < 1000
    limit, capped = tolerance(lambda d: (d * 2, 0), 10, 20)
    assert math.isclose(limit, 5, abs_tol=1e-8) and not capped
    assert tolerance(lambda _: (0, 0), 10, 20) == (20, True)
    for parameter, expected in (
        (0, math.degrees(2 * math.asin(100 / 10_000))),
        (1, tilt - math.degrees(math.atan2(35, 5100))),
        (4, 0.7),
    ):
        error = partial(
            displacement, parameter, camera, angles, pixel, flat, 5000, 30_000
        )
        limit, capped = tolerance(error, 100, 5)
        assert math.isclose(limit, expected, rel_tol=1e-8) and not capped
    print("Geometry checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Run the analytical checks only"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data/08_reporting"
    )
    parser.add_argument(
        "--distances", type=float, nargs="+", default=[500, 1000, 2000, 5000, 10_000]
    )
    parser.add_argument("--heights", type=float, nargs="+", default=[15, 35, 100])
    parser.add_argument("--fovs", type=float, nargs="+", default=[54.2, 87])
    parser.add_argument(
        "--angular-errors", type=float, nargs="+", default=[0.01, 0.1, 1]
    )
    parser.add_argument("--position-errors", type=float, nargs="+", default=[0.1, 1, 5])
    parser.add_argument(
        "--precision-caps",
        type=float,
        nargs=2,
        default=[5, 200],
        metavar=("DEG", "METRES"),
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100,
        help="Example horizontal error budget in metres",
    )
    parser.add_argument(
        "--max-range", type=float, default=30_000, help="Horizontal range (m)"
    )
    args = parser.parse_args()
    if args.check:
        check()
        return
    positive = (
        args.distances
        + args.heights
        + args.angular_errors
        + args.position_errors
        + args.precision_caps
        + [args.budget, args.max_range]
    )
    if not all(math.isfinite(value) and value > 0 for value in positive):
        parser.error(
            "Distances, heights, errors, caps, budget and range must be finite and > 0"
        )
    if not all(math.isfinite(value) and 1 < value < 179 for value in args.fovs):
        parser.error("FOV must be finite and between 1 and 179 degrees")
    if (
        max(args.distances) >= args.max_range
        or max(args.angular_errors) >= min(args.fovs)
        or max(args.fovs) + max(args.angular_errors) >= 180
    ):
        parser.error(
            "Range must exceed distances; perturbed FOV must stay within (0, 180)"
        )
    names = [
        "azimuth",
        "tilt",
        "roll",
        "hfov",
        "height",
        "position_right",
        "position_forward",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with (
        (args.output / "errors.csv").open("w", newline="") as errors_file,
        (args.output / "precision.csv").open("w", newline="") as precision_file,
    ):
        errors, precision = csv.writer(errors_file), csv.writer(precision_file)
        fields = [
            "terrain",
            "distance_m",
            "height_m",
            "hfov_deg",
            "u",
            "v",
            "parameter",
            "unit",
        ]
        errors.writerow(fields + ["delta", "status", "along_m", "cross_m", "error_m"])
        precision.writerow(fields + ["budget_m", "limit", "capped", "status"])
        cases = itertools.product(
            args.distances,
            args.heights,
            args.fovs,
            (0.1, 0.5, 0.9),
            (0.1, 0.5, 0.9),
            (0, 0.02, 0.1),
        )
        count, missed, occluded = 0, 0, 0
        for distance, height, hfov, u, v, slope in cases:
            terrain = [
                (-args.max_range, 0),
                (distance / 2, 0),
                (args.max_range, slope * (args.max_range - distance / 2)),
            ]
            label = f"rising_{slope:g}"
            camera = (0, 0, height)
            target_z = elevation_at(terrain, distance)
            status = "visible"
            try:
                azimuth, tilt = pose(
                    u, v, hfov, math.atan2(target_z - camera[2], distance)
                )
                angles, pixel = (azimuth, tilt, 0, hfov), (u, v)
                baseline = intersect(
                    ray(*pixel, *angles), camera, terrain, args.max_range
                )
                if (
                    baseline is None
                    or math.hypot(baseline[0], baseline[1] - distance) >= 0.01
                ):
                    status = "invalid_baseline"
            except ValueError:
                status = "unreachable_pixel"
            if status != "visible":
                occluded += 1
            for parameter, name in enumerate(names):
                unit = "deg" if parameter < 4 else "m"
                prefix = [label, distance, height, hfov, u, v, name, unit]

                if status != "visible":
                    precision.writerow(prefix + [args.budget, "", "", status])
                    continue
                error = partial(
                    displacement,
                    parameter,
                    camera,
                    angles,
                    pixel,
                    terrain,
                    distance,
                    args.max_range,
                )
                deltas = args.angular_errors if parameter < 4 else args.position_errors
                for delta, sign in itertools.product(deltas, (-1, 1)):
                    shift = error(delta * sign)
                    count += 1
                    if shift is None:
                        missed += 1
                        errors.writerow(prefix + [delta * sign, "no_hit", "", "", ""])
                    else:
                        cross, along = shift
                        errors.writerow(
                            prefix
                            + [delta * sign, "hit", along, cross, math.hypot(*shift)]
                        )
                cap = args.precision_caps[parameter >= 4]
                if parameter == 3:
                    cap = min(cap, hfov / 2, (180 - hfov) / 2)
                limit, capped = tolerance(error, args.budget, cap)
                precision.writerow(prefix + [args.budget, limit, capped, "visible"])
    config = dict(
        vars(args),
        python=sys.version,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with (args.output / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2, default=str)
    print(
        f"{count} perturbations; {missed} no-hit results; "
        f"{occluded} invalid baseline cases. Output: {args.output}"
    )


if __name__ == "__main__":
    main()

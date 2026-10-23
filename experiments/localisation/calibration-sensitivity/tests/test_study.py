import itertools
import math
from functools import partial

from study import displacement, intersect, pose, ray, tolerance


def test_geometry():
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

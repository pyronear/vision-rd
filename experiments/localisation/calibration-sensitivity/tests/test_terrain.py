import math

import numpy as np
import pytest
from smoke_localisation.ray import curvature_drop_m

from terrain import GEOD, sweep_view

SITE = dict(
    site="test", support_id="test", latitude=44.2, longitude=-0.7, mast_height_m=40
)


def flat_dsm():
    return (
        np.zeros((101, 101)),
        np.linspace(44.4, 44, 101),
        np.linspace(-0.95, -0.45, 101),
    )


def test_curved_flat_control_and_tilt_sensitivity():
    view, errors, precision = sweep_view(SITE, flat_dsm(), 0, 5000, 100)
    assert view["status"] == "visible" and view["nominal_offset_m"] < 1e-3
    case = next(c for c in errors if c["parameter"] == "tilt" and c["delta"] == -0.1)
    elevation = math.atan2(-40 - curvature_drop_m(5000), 5000) + math.radians(0.1)
    # Independent quadratic intersection with the effective-Earth parabola.
    a = curvature_drop_m(1)
    b = math.tan(elevation)
    distance = 80 / (-b + math.sqrt(b * b - 4 * a * 40))
    assert math.isclose(case["error_m"], distance - 5000, abs_tol=1e-3)
    assert case["error_m"] > 1000
    tilt = next(c for c in precision if c["parameter"] == "tilt")
    assert tilt["tested_limit"] == 0.003 and tilt["limit_status"] == "sampled_only"


def test_missing_projection_is_not_zero_error():
    _, errors, precision = sweep_view(SITE, flat_dsm(), 0, 10_000, 100)
    missing = [c for c in errors if c["status"] == "no_hit"]
    assert missing and all(c["error_m"] is None for c in missing)
    tilt = next(c for c in precision if c["parameter"] == "tilt")
    assert tilt["tested_limit"] < 0.1


def test_intervening_ridge_excludes_target():
    elev, lats, lons = flat_dsm()
    _, ridge_lat, _ = GEOD.fwd(SITE["longitude"], SITE["latitude"], 0, 1000)
    elev[np.abs(lats - ridge_lat) < 0.004] = 80
    view, errors, precision = sweep_view(SITE, (elev, lats, lons), 0, 5000, 100)
    assert view["status"] == "occluded_or_unresolved"
    assert view["nominal_offset_m"] > 1000 and not errors and not precision


def test_uncovered_target_fails_explicitly():
    elev, lats, lons = flat_dsm()
    with pytest.raises(ValueError, match="DSM does not cover"):
        sweep_view(SITE, (elev[:2], lats[:2], lons), 0, 5000, 100)


def test_horizontal_position_does_not_follow_surface_height():
    elev, lats, lons = flat_dsm()
    elev[:] = (lons - SITE["longitude"]) * 100_000
    _, errors, _ = sweep_view(SITE, (elev, lats, lons), 0, 5000, 100)
    case = next(
        c
        for c in errors
        if c["parameter"] == "position_east"
        and c["delta"] == 1
        and c["u"] == c["v"] == 0.5
    )
    # Follow terrain laterally without moving the camera vertically.
    assert case["error_m"] > 100

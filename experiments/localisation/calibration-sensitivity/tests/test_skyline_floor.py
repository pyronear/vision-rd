from types import SimpleNamespace

import numpy as np
import pytest
import smoke_localisation.calibration as calibration
from smoke_localisation.camera import pixel_to_ray, vfov_deg
from smoke_localisation.dem import dsm_interpolator
from smoke_localisation.sky import find_horizon_line

import skyline_floor as floor


def test_combined_trials_isolate_the_skyline_cutoff():
    cases = list(floor.cases())
    limited = [d for name, d in cases if name == "combined"]
    full = [d for name, d in cases if name == "combined_no_cutoff"]
    assert len(limited) == len(full) == 24
    for before, after in zip(limited, full, strict=True):
        assert before["extent"] == 25_000
        assert after["extent"] == floor.EXTENT
        assert {k: v for k, v in before.items() if k != "extent"} == {
            k: v for k, v in after.items() if k != "extent"
        }


def test_render_matches_rotated_camera_rays():
    angles = np.linspace(-40, 40, 801)
    horizon = 0.03 * angles
    mask = floor.render(horizon, angles, (72, 128), 60, 2, 1)
    for row, col in ((30, 20), (35, 90), (39, 60), (50, 10)):
        az, el = pixel_to_ray(
            (col + 0.5) / 128, (row + 0.5) / 72, 180, 60, (128, 72), 2, 1
        )
        assert bool(mask[row, col]) == (el > 0.03 * (az - 180))


def test_one_pixel_edge_error_has_known_signed_tilt_cost(monkeypatch):
    monkeypatch.setattr(calibration, "dem_skyline", lambda *a, **k: np.zeros_like(a[3]))
    dsm = np.zeros((5, 5)), np.linspace(44.1, 43.9, 5), np.linspace(4.9, 5.1, 5)
    mask = floor.render(np.zeros(2), [-40, 40], (360, 640), 60, 2, 0)
    results = [
        calibration.calibrate(m, 44, 5, 20, 180, 60, *dsm)
        for m in (mask, floor.shifted(mask, 1))
    ]
    assert abs(results[0].tilt_deg - 2) < vfov_deg(60, (640, 360)) / 360
    assert results[1].tilt_deg - results[0].tilt_deg == pytest.approx(
        -vfov_deg(60, (640, 360)) / 360
    )
    assert not floor.shifted(mask, -360).any()  # No sky wraps from the top edge.


def test_column_shifts_move_only_their_own_skyline():
    mask = np.tile([1, 1, 1, 0, 0, 0], (4, 1)).T.astype(np.float32)
    moved = floor.shifted(mask, np.array([1, -1, 0, 2]))
    assert find_horizon_line(moved).tolist() == [3, 1, 2, 4]
    assert np.all(np.diff(moved, axis=0) <= 0)


def test_column_noise_preserves_block_correlation_and_seed():
    offsets = floor.column_noise(131, 2, 16, 4)
    assert np.array_equal(offsets, floor.column_noise(131, 2, 16, 4))
    assert offsets.shape == (131,)
    assert np.issubdtype(offsets.dtype, np.integer)
    assert np.unique(offsets).size > 1
    for block in np.array_split(offsets[:128], 8):
        assert np.ptp(block) == 0
    assert np.ptp(offsets[128:]) == 0


def test_balanced_column_errors_average_out_but_common_bias_does_not(monkeypatch):
    monkeypatch.setattr(calibration, "dem_skyline", lambda *a, **k: np.zeros_like(a[3]))
    dsm = np.zeros((5, 5)), np.linspace(44.1, 43.9, 5), np.linspace(4.9, 5.1, 5)
    mask = floor.render(np.zeros(2), [-40, 40], (216, 384), 60, 2, 0)
    alternating = np.tile([-1, 1], 192)
    results = [
        calibration.calibrate(m, 44, 5, 20, 180, 60, *dsm, max_azimuth_offset_deg=0)
        for m in (mask, floor.shifted(mask, alternating), floor.shifted(mask, 1))
    ]
    assert abs(results[1].tilt_deg - results[0].tilt_deg) < 0.001
    assert results[2].tilt_deg - results[0].tilt_deg == pytest.approx(
        -vfov_deg(60, (384, 216)) / 216
    )


def test_dsm_defects_keep_absolute_camera_altitude_and_rejections(monkeypatch):
    monkeypatch.setattr(floor, "dem_skyline", lambda *a, **k: np.zeros_like(a[3]))
    monkeypatch.setattr(
        floor,
        "cases",
        lambda *_: iter(
            [
                ("perfect", {}),
                ("noise", dict(noise=4, seed=0)),
                ("canopy", dict(canopy=5)),
                ("mask", dict(small=True)),
            ]
        ),
    )
    altitudes = []

    def fitted(mask, lat, lon, height, az, fov, *dsm, **kwargs):
        assert mask.ndim == 2 and np.isfinite(mask).all()
        assert mask.min() >= 0 and mask.max() <= 1
        altitudes.append(float(dsm_interpolator(*dsm)((lat, lon))) + height)
        if len(altitudes) == 3:
            raise ValueError("bad fit")
        return SimpleNamespace(tilt_deg=2, azimuth_deg=az, residual_deg=0)

    monkeypatch.setattr(floor, "calibrate", fitted)
    cam = dict(
        lat=44,
        lon=5,
        mast_height_m=20,
        pose_azimuth_deg=180,
        angle_of_view_deg_platform=60,
    )
    dsm = np.zeros((5, 5)), np.linspace(44.1, 43.9, 5), np.linspace(4.9, 5.1, 5)
    rows = list(floor.evaluate(cam, dsm, (720, 1280)))
    assert altitudes == pytest.approx([20, 20, 20, 20])
    assert rows[2]["status"] == "rejected" and rows[2]["reason"] == "bad fit"
    assert rows[3]["status"] == "accepted"

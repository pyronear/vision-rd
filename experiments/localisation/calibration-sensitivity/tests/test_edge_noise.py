from pathlib import Path

import numpy as np
import pytest
import requests
import smoke_localisation.calibration as calibration
import yaml
from smoke_localisation.camera import vfov_deg

import edge_noise


def test_offline_run_uses_real_fitter_without_network_and_preserves_sampler(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline controls must not request network access")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    original = calibration.dem_skyline
    cfg = yaml.safe_load(Path("edge_noise.yaml").read_text())
    cfg.update(
        shape=[216, 384],
        profiles=["flat"],
        rolls_deg=[0],
        azimuth_search_deg={"fixed": 0},
        noise_stddev_px=[1],
        block_widths_px=[1],
        seed_count=2,
    )
    rows = list(edge_noise.run(cfg))
    assert calibration.dem_skyline is original
    assert len(rows) == 5
    assert all(r["status"] == "accepted" for r in rows)
    for row in rows:
        if row["scenario"] == "common_bias":
            assert row["delta_tilt_deg"] == pytest.approx(
                -row["bias_px"] * vfov_deg(60, (384, 216)) / 216
            )
            assert row["mean_edge_shift_px"] == row["bias_px"]
    assert rows[0]["delta_tilt_deg"] == 0


def test_summary_keeps_rejections_in_trial_counts_and_uses_absolute_truth():
    shared = dict(
        profile="flat",
        roll_deg=0,
        fit="fixed",
        scenario="noise",
        sigma_px=1,
        block_px=1,
    )
    trials = [
        dict(shared, status="accepted", tilt_error_deg=error, delta_tilt_deg=0)
        for error in (-0.02, 0.005, 0.02)
    ]
    trials.append(dict(shared, status="rejected", reason="bad fit"))
    (row,) = edge_noise.summarize(trials)
    assert (row["trials"], row["accepted"]) == (4, 3)
    assert row["within_0p009_deg"] == 1
    assert row["within_0p03_deg"] == 3
    assert row["max_abs_tilt_error_deg"] == 0.02
    assert row["max_abs_delta_tilt_deg"] == 0
    assert row["std_tilt_error_deg"] == pytest.approx(np.std([-0.02, 0.005, 0.02]))


def test_all_rejected_group_has_no_accuracy_score():
    (row,) = edge_noise.summarize(
        [
            dict(
                profile="ridges",
                roll_deg=1,
                fit="joint",
                scenario="noise",
                sigma_px=2,
                block_px=128,
                status="rejected",
                reason="boundary",
            )
        ]
    )
    assert row["accepted"] == 0
    assert row["max_abs_tilt_error_deg"] is None
    assert row["std_tilt_error_deg"] is None
    assert row["within_0p009_deg"] == 0

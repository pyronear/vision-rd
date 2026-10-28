import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import repeatability
from repeatability import epochs, run, sigma, summarize, tilt_requirement

CFG = yaml.safe_load((Path(__file__).parents[1] / "repeatability.yaml").read_text())


def frames(values):
    return [dict(tilt_deg=v, tilt_uncertainty_deg=0.05) for v in values]


def test_single_bad_frame_is_not_a_camera_move():
    values = [1.0] * 25
    values[12] = 4
    assert len(epochs(frames(values), CFG)) == 1
    assert sigma(values) == 0


def test_persistent_camera_move_is_not_counted_as_random_noise():
    groups = epochs(frames([1.0] * 12 + [2.1] * 12), CFG)
    assert [len(g) for g in groups] == [12, 12]
    assert all(sigma([r["tilt_deg"] for r in group]) == 0 for group in groups)


def test_short_changed_state_is_retained_separately():
    groups = epochs(frames([1.0] * 3 + [2.1] * 12), CFG)
    assert [len(g) for g in groups] == [3, 12]


def test_temporary_changed_state_is_retained():
    groups = epochs(frames([1.0] * 16 + [2.0] * 2 + [1.0] * 2), CFG)
    assert [len(g) for g in groups] == [16, 2, 2]


def test_flat_range_requires_tight_tilt():
    dsm = (
        np.zeros((101, 101)),
        np.linspace(44.4, 44, 101),
        np.linspace(-0.95, -0.45, 101),
    )
    target = tilt_requirement(dict(lat=44.2, lon=-0.7), 40, 0, dsm, 5000, CFG)
    assert target == 0.003


def test_all_rejected_is_not_zero_uncertainty(tmp_path):
    rows = [
        dict(
            pose="P01",
            pose_id="raw",
            camera="C01",
            height_known=False,
            status="rejected",
            reason="cloud",
        )
        for t in (True, False)
    ]
    (tmp_path / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "profiles.json").write_text(
        json.dumps(dict(raw=dict(hfov=87, fov_profile_complete=False)))
    )
    summarize(SimpleNamespace(output=tmp_path), CFG)
    result = next(csv.DictReader((tmp_path / "summary.csv").open()))
    assert result["accepted"] == "0"
    assert result["repeatability_sigma_deg"] == result["absolute_bias_deg"] == ""
    assert result["requirement_2000m_status"] == "no_reference_fit"


def test_manifest_cannot_escape_bundle(tmp_path):
    (tmp_path / "cameras.csv").write_text("pose_id,platform_camera\np,C\n")
    (tmp_path / "images.csv").write_text("pose_id,file\np,../outside.jpg\n")
    with pytest.raises(ValueError, match="escapes"):
        run(SimpleNamespace(input=tmp_path), CFG)


def test_resume_keeps_one_record_and_failed_scenario_is_unknown(tmp_path, monkeypatch):
    (tmp_path / "cameras.csv").write_text(
        "pose_id,platform_camera,lat,lon,mast_height_m,angle_of_view_deg_platform\n"
        "p,C,44.2,-.7,15,60\n"
    )
    (tmp_path / "images.csv").write_text(
        "pose_id,file,datetime_utc\np,one.jpg,2024-01-01T12:00:00Z\n"
    )
    (tmp_path / "one.jpg").write_bytes(b"fixture")
    monkeypatch.setattr(repeatability.cv2, "imread", lambda *a: np.zeros((32, 64, 3)))
    monkeypatch.setattr(
        repeatability,
        "detect_sky",
        lambda *a: np.vstack((np.ones((16, 64)), np.zeros((16, 64)))),
    )
    monkeypatch.setattr(repeatability, "load_dsm", lambda *a: (0, 0, 0))
    monkeypatch.setattr(repeatability, "tilt_requirement", lambda *a: None)

    def fitted(*args, **kwargs):
        if kwargs.get("threshold") == CFG["mask_threshold_scenarios"][0]:
            raise ValueError("unsupported threshold")
        return SimpleNamespace(
            tilt_deg=0.4,
            roll_deg=0,
            azimuth_deg=0,
            residual_deg=0.01,
            tilt_uncertainty_deg=0.05,
            azimuth_uncertainty_deg=1,
        )

    monkeypatch.setattr(repeatability, "fit", fitted)
    args = SimpleNamespace(
        input=tmp_path,
        output=tmp_path / "out",
        cache=tmp_path / "cache",
        params=Path(__file__).parents[1] / "repeatability.yaml",
    )
    run(args, CFG)
    run(args, CFG)
    assert len((args.output / "frames.jsonl").read_text().splitlines()) == 1
    profile = json.loads((args.output / "profiles.json").read_text())["p"]
    assert profile["mask_sensitivity_deg"] is None
    assert profile["mask_failures"] == ["unsupported threshold"]

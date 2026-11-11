"""Offline skyline-edge noise controls for #163: no images, terrain or hardware."""

import argparse
import csv
import hashlib
import itertools
import json
from importlib.metadata import distribution, version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import smoke_localisation.calibration as calibration
import yaml
from smoke_localisation.sky import SKY_INPUT_SIZE, find_horizon_line

from skyline_floor import column_noise, render, shifted


def profile(angles, name, ridge):
    """Analytic elevation in degrees, indexed by relative azimuth in degrees."""
    angles = np.asarray(angles)
    if name == "flat":
        return np.zeros_like(angles)
    if name != "ridges":
        raise ValueError(f"Unknown skyline profile: {name}")
    return (
        ridge["sine_amplitude"] * np.sin(ridge["sine_frequency"] * angles)
        + ridge["cosine_amplitude"] * np.cos(ridge["cosine_frequency"] * angles)
        + ridge["peak_amplitude"]
        * np.exp(-(((angles - ridge["peak_center"]) / ridge["peak_width"]) ** 2))
    )


def cases(width, cfg):
    yield (
        dict(scenario="clean", sigma_px=0, block_px=0, seed=-1),
        np.zeros(width, dtype=int),
    )
    for bias in cfg["common_bias_px"]:
        yield (
            dict(
                scenario="common_bias",
                sigma_px=0,
                block_px=width,
                seed=-1,
                bias_px=bias,
            ),
            np.full(width, bias, dtype=int),
        )
    for sigma, block, seed in itertools.product(
        cfg["noise_stddev_px"], cfg["block_widths_px"], range(cfg["seed_count"])
    ):
        yield (
            dict(scenario="noise", sigma_px=sigma, block_px=block, seed=seed),
            column_noise(width, sigma, block, seed),
        )


def run(cfg):
    h, w = cfg["shape"]
    # Only the interpolator scaffold is used: the DSM skyline sampler below is
    # replaced by the exact analytic profile. No geographic terrain is evaluated.
    dsm = np.zeros((5, 5)), np.linspace(0.1, -0.1, 5), np.linspace(-0.1, 0.1, 5)
    angles = np.arange(-90.0, 90.025, 0.025)
    for name, roll in itertools.product(cfg["profiles"], cfg["rolls_deg"]):
        clean = render(
            profile(angles, name, cfg["ridge_profile"]),
            angles,
            (h, w),
            cfg["hfov_deg"],
            cfg["tilt_deg"],
            roll,
        )
        reference = find_horizon_line(clean)

        def skyline(_lat, _lon, _z, bearings, _interp, name=name, **_kwargs):
            return profile(bearings - 180.0, name, cfg["ridge_profile"])

        with patch.object(calibration, "dem_skyline", skyline):
            for mode, search in cfg["azimuth_search_deg"].items():
                baseline = None
                for case, offsets in cases(w, cfg):
                    mask = shifted(clean, offsets)
                    edge = find_horizon_line(mask)
                    valid = np.flatnonzero((edge >= 0) & (edge < h - 1))
                    selected = valid[
                        np.linspace(
                            0,
                            len(valid) - 1,
                            min(len(valid), SKY_INPUT_SIZE),
                            dtype=int,
                        )
                    ]
                    row = dict(
                        profile=name,
                        roll_deg=roll,
                        fit=mode,
                        **case,
                        mean_edge_shift_px=float(np.mean(edge - reference)),
                        sampled_mean_edge_shift_px=float(
                            np.mean((edge - reference)[selected])
                        )
                        if selected.size
                        else None,
                    )
                    try:
                        result = calibration.calibrate(
                            mask,
                            0,
                            0,
                            20,
                            180,
                            cfg["hfov_deg"],
                            *dsm,
                            max_azimuth_offset_deg=search,
                            max_skyline_distance_m=50_000,
                        )
                        if case["scenario"] == "clean":
                            baseline = result.tilt_deg
                        row.update(
                            status="accepted",
                            tilt_error_deg=result.tilt_deg - cfg["tilt_deg"],
                            delta_tilt_deg=result.tilt_deg - baseline
                            if baseline is not None
                            else None,
                            azimuth_error_deg=(result.azimuth_deg - 180 + 180) % 360
                            - 180,
                            residual_deg=result.residual_deg,
                        )
                    except ValueError as exc:
                        row.update(status="rejected", reason=str(exc))
                    yield row


def summarize(records):
    groups = {}
    for row in records:
        key = tuple(
            row.get(k, 0)
            for k in (
                "profile",
                "roll_deg",
                "fit",
                "scenario",
                "sigma_px",
                "block_px",
                "bias_px",
            )
        )
        groups.setdefault(key, []).append(row)
    for key, trials in groups.items():
        accepted = [r for r in trials if r["status"] == "accepted"]
        errors = np.array([r["tilt_error_deg"] for r in accepted])
        deltas = [
            r["delta_tilt_deg"] for r in accepted if r["delta_tilt_deg"] is not None
        ]
        yield dict(
            zip(
                (
                    "profile",
                    "roll_deg",
                    "fit",
                    "scenario",
                    "sigma_px",
                    "block_px",
                    "bias_px",
                ),
                key,
                strict=True,
            ),
            trials=len(trials),
            accepted=len(accepted),
            mean_tilt_error_deg=float(np.mean(errors)) if errors.size else None,
            std_tilt_error_deg=float(np.std(errors)) if errors.size else None,
            max_abs_tilt_error_deg=float(np.max(np.abs(errors)))
            if errors.size
            else None,
            max_abs_delta_tilt_deg=max(map(abs, deltas)) if deltas else None,
            within_0p009_deg=int(np.sum(np.abs(errors) <= 0.009)),
            within_0p03_deg=int(np.sum(np.abs(errors) <= 0.03)),
            within_0p1_deg=int(np.sum(np.abs(errors) <= 0.1)),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("edge_noise.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/08_reporting/edge-noise-offline")
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.params.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with (args.output / "trials.jsonl").open("w", buffering=1) as stream:
        for row in run(cfg):
            records.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if len(records) % 100 == 0:
                print(f"Completed {len(records)} trials", flush=True)
    summary = list(summarize(records))
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    provenance = dict(
        config=cfg,
        sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("skyline_floor.py"),
                args.params,
            )
        },
        runtime=json.loads(
            distribution("smoke-localisation").read_text("direct_url.json")
        ),
        runtime_calibration_sha256=hashlib.sha256(
            Path(calibration.__file__).read_bytes()
        ).hexdigest(),
        versions={p: version(p) for p in ("numpy", "scipy", "opencv-python-headless")},
        method="analytic skyline sampler; unchanged calibration fitter and gates",
    )
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Saved {len(records)} trials and {len(summary)} summaries to {args.output}")


if __name__ == "__main__":
    main()

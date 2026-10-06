# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Screen one camera view and choose the next measurement or calibration task."""

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

from study import intersect, project, ray

SOURCES = {
    "azimuth_deg": (0,),
    "tilt_deg": (1,),
    "roll_deg": (2,),
    "hfov_deg": (3,),
    "vfov_deg": (4,),
    "relative_height_m": (5,),
    "position_m": (6, 7),
    "point_px": (8, 9),
    "terrain_slope": (10,),
}
ACTIONS = {
    "azimuth_deg": "Recalibrate direction against known landmarks or the sun.",
    "tilt_deg": "Reduce vertical pointing residuals after preset returns.",
    "roll_deg": "Level or recalibrate image roll at this preset.",
    "hfov_deg": "Verify horizontal field of view at this zoom and capture resolution.",
    "vfov_deg": "Verify vertical field of view at this zoom and capture resolution.",
    "relative_height_m": "Improve relative elevation data, including terrain error.",
    "position_m": "Check the camera's horizontal coordinates.",
    "point_px": "Improve ground-origin point selection or image scale.",
    "terrain_slope": "Check the terrain profile and its slope near the target.",
}


LIMITS = {
    "range_m": (1, 100_000),
    "max_range_m": (1, 100_000),
    "height_above_target_m": (-20_000, 20_000),
    "terrain_slope": (-10, 10),
    "tilt_deg": (-90, 90),
    "roll_deg": (-180, 180),
    "bearing_offset_deg": (-180, 180),
    "hfov_deg": (0, 180),
    "vfov_deg": (0, 180),
    "width_px": (1, 100_000),
    "height_px": (1, 100_000),
    "budget_m": (0.001, 100_000),
}


def validate(view):
    metadata = {"name", "evidence", "ground_origin_visible", "bounds"}
    if not isinstance(view, dict) or set(view) != set(LIMITS) | metadata:
        raise ValueError("Profile fields must match view.example.json")
    for key, (low, high) in LIMITS.items():
        value = view[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"{key} must be a finite number in [{low}, {high}]")
    if view["range_m"] >= view["max_range_m"]:
        raise ValueError("Range must be below the horizontal analysis limit")
    if any(int(view[k]) != view[k] for k in ("width_px", "height_px")):
        raise ValueError("Image dimensions must be whole pixels")
    if (
        not isinstance(view["name"], str)
        or not view["name"].strip()
        or view["evidence"] not in ("assumed", "measured")
        or type(view["ground_origin_visible"]) is not bool
    ):
        raise ValueError(
            "Declare a name, assumed/measured evidence and ground visibility"
        )
    bounds = view["bounds"]
    if not isinstance(bounds, dict) or set(bounds) != set(SOURCES):
        raise ValueError("Error fields must match the example")
    # ponytail: local angular screen; full uncertainty propagation belongs in #113.
    for key, value in bounds.items():
        cap = 5 if key.endswith("_deg") else 100_000
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= cap
        ):
            raise ValueError(f"{key} uncertainty must be finite and in [0, {cap}]")
    if not any(bounds.values()):
        raise ValueError("Supply a positive uncertainty; exact inputs are not a plan")
    if abs(view["terrain_slope"]) + bounds["terrain_slope"] > 10:
        raise ValueError("Slope uncertainty exceeds the local plane's domain")
    if any(
        not 0 < view[k] - bounds[k] <= view[k] + bounds[k] < 180
        for k in ("hfov_deg", "vfov_deg")
    ):
        raise ValueError("Both FOV error ranges must stay within (0, 180)")


def stress(view, pixel, active):
    """Test lower, nominal and upper inputs; this is not a continuous error bound."""
    levels = [(0.0,)] * 11
    for source in active:
        bound = view["bounds"][source]
        for axis in SOURCES[source]:
            levels[axis] = (-bound, 0.0, bound) if bound else (0.0,)
    maximum, missing, count = 0.0, 0, 0
    for a, t, r, h, v, z, x, y, px, py, slope_delta in itertools.product(*levels):
        u, w = pixel[0] + px / view["width_px"], pixel[1] + py / view["height_px"]
        slope = view["terrain_slope"] + slope_delta
        distance, limit = view["range_m"], view["max_range_m"]
        terrain = [
            (-limit, slope * (-limit - distance)),
            (limit, slope * (limit - distance)),
        ]
        direction = ray(
            u,
            w,
            -view["bearing_offset_deg"] + a,
            view["tilt_deg"] + t,
            view["roll_deg"] + r,
            view["hfov_deg"] + h,
            view["vfov_deg"] + v,
        )
        hit = (
            intersect(
                direction, (x, y, view["height_above_target_m"] + z), terrain, limit
            )
            if 0 <= u <= 1 and 0 <= w <= 1
            else None
        )
        count += 1
        if hit is None:
            missing += 1
        else:
            maximum = max(maximum, math.hypot(hit[0], hit[1] - distance))
    return {"max_valid_error_m": maximum, "missing_hits": missing, "cases": count}


def assess(view):
    validate(view)
    report = {
        "name": view["name"],
        "evidence": view["evidence"],
        "budget_m": view["budget_m"],
        "model": "sampled local plane; no continuous bound or field acceptance",
    }
    if not view["ground_origin_visible"]:
        return report | {
            "decision": "GROUND_CONSTRAINT",
            "next": "Establish the ground origin or obtain another view first.",
        }
    distance, height = view["range_m"], view["height_above_target_m"]
    pixel = project(
        (0, distance, -height),
        -view["bearing_offset_deg"],
        view["tilt_deg"],
        view["roll_deg"],
        view["hfov_deg"],
        view["vfov_deg"],
    )
    if pixel is None or stress(view, pixel, ())["missing_hits"]:
        return report | {
            "decision": "CHANGE_VIEW",
            "next": "Choose a preset with a visible ground target inside the image.",
        }
    drivers = {source: stress(view, pixel, (source,)) for source in SOURCES}
    joint = stress(view, pixel, SOURCES)
    # Curvature screening: nominal line against a parabolic Earth; no refraction.
    c = height + view["terrain_slope"] * distance
    b, a = -c / distance, 1 / (2 * 6_371_000)
    discriminant = b * b - 4 * a * c
    bias = (
        2 * c / (-b + math.sqrt(discriminant)) - distance
        if c > 0 and discriminant > 0
        else None
    )
    ranking = sorted(
        drivers,
        key=lambda k: (
            drivers[k]["missing_hits"] > 0,
            drivers[k]["max_valid_error_m"],
        ),
        reverse=True,
    )
    first = ranking[0]
    blockers = [
        k
        for k in ranking
        if drivers[k]["missing_hits"]
        or drivers[k]["max_valid_error_m"] >= view["budget_m"]
    ]
    if bias is None or bias >= view["budget_m"]:
        blockers.insert(0, "curvature_geometry")
    if joint["missing_hits"]:
        decision, action = (
            "UNSTABLE",
            "Improve measured inputs or change view: tested projections are missing.",
        )
    elif bias is None or bias >= view["budget_m"]:
        decision, action = (
            "GEOMETRY_REQUIRED",
            "Use curvature-aware terrain geometry before judging this accuracy target.",
        )
    elif joint["max_valid_error_m"] + bias > view["budget_m"]:
        decision, action = "REDUCE_UNCERTAINTY", ACTIONS[first]
    elif view["evidence"] == "assumed":
        decision, action = (
            "MEASURE_VIEW",
            "Verify the view/capture profile, then measure " + first + " residuals.",
        )
    else:
        decision, action = (
            "FIELD_CHECK",
            "Test known ground points and real fires before accepting this view.",
        )
    return report | {
        "decision": decision,
        "next": action,
        "pixel": pixel,
        "joint": joint,
        "curvature_bias_estimate_m": bias,
        "drivers": drivers,
        "priority": ranking,
        "individual_blockers": blockers,
        "screening_total_m": None
        if bias is None or joint["missing_hits"]
        else joint["max_valid_error_m"] + bias,
    }


def check():
    view = json.loads(Path(__file__).with_name("view.example.json").read_text())
    result = assess(view)
    assert result["decision"] == "MEASURE_VIEW" and result["screening_total_m"] < 250
    far = view | {"range_m": 10_000}
    assert assess(far)["decision"] == "REDUCE_UNCERTAINTY"
    flat = copy.deepcopy(view)
    flat.update(height_above_target_m=35, tilt_deg=0)
    result = assess(flat)
    assert (
        result["decision"] == "GEOMETRY_REQUIRED"
        and result["drivers"]["point_px"]["max_valid_error_m"] > 500
    )
    assert {"point_px", "curvature_geometry"} <= set(result["individual_blockers"])
    flat["bounds"]["tilt_deg"] = 1
    unstable = assess(flat)
    assert unstable["decision"] == "UNSTABLE" and unstable["screening_total_m"] is None
    view["ground_origin_visible"] = False
    assert assess(view)["decision"] == "GROUND_CONSTRAINT"
    view["ground_origin_visible"], view["evidence"] = True, "measured"
    assert assess(view)["decision"] == "FIELD_CHECK"
    view["tilt_deg"] = 60
    assert assess(view)["decision"] == "CHANGE_VIEW"
    for a, t, r in itertools.product((-30, 0, 30), (-15, 0, 15), (-5, 0, 5)):
        direction = ray(0.3, 0.6, a, t, r, 54.2, 41.7)
        pixel = project(direction, a, t, r, 54.2, 41.7)
        assert pixel is not None and math.dist(pixel, (0.3, 0.6)) < 1e-12
    bad = copy.deepcopy(view)
    for key, value in (("point_px", -1), ("azimuth_deg", 360), ("point_px", math.nan)):
        bad["bounds"][key] = value
        try:
            assess(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid bounds accepted")
        bad = copy.deepcopy(view)
    print("Planning checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("view.example.json"),
    )
    parser.add_argument(
        "--range-m",
        type=float,
        nargs="+",
        help="Assess these nominal planar ranges with the same preset",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Include the complete input profile in machine-readable output",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    try:
        profile = json.loads(args.profile.read_text())
        plans = []
        for distance in args.range_m or [profile["range_m"]]:
            view = profile | {"range_m": distance}
            plans.append({"input": view, "result": assess(view)})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(plans, indent=2, allow_nan=False))
        return
    for plan in plans:
        view, result = plan["input"], plan["result"]
        print(
            f"\n{result['name']} — {view['range_m']:g} m / {result['evidence']} inputs"
        )
        print(
            f"Decision: {result['decision'].replace('_', ' ').lower()}\n"
            f"Next: {result['next']}"
        )
        if "joint" in result:
            joint = result["joint"]
            print(
                f"Budget {result['budget_m']:g} m; "
                f"{joint['missing_hits']} missing hits / {joint['cases']} cases."
            )
            if result["screening_total_m"] is not None:
                print(
                    f"Screening total: {result['screening_total_m']:.1f} m "
                    "(largest tested shift + curvature estimate)."
                )
            bias = result["curvature_bias_estimate_m"]
            print(
                "Estimated omitted-curvature bias: "
                + ("no nominal intersection" if bias is None else f"{bias:.1f} m")
            )
            print(
                "Separate source tests: "
                + ", ".join(
                    f"{k}: "
                    + (
                        "missing projection"
                        if result["drivers"][k]["missing_hits"]
                        else f"{result['drivers'][k]['max_valid_error_m']:.1f} m"
                    )
                    for k in result["priority"][:3]
                )
            )
            if result["individual_blockers"]:
                print(
                    "Individual budget blockers: "
                    + ", ".join(result["individual_blockers"])
                )
    print(
        "\nPlanning screen: sampled inputs on a local plane; no continuous bound "
        "or field acceptance. Refraction, distortion and unseen ridges are excluded."
    )


if __name__ == "__main__":
    main()

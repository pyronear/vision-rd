# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Plot committed skyline-edge summaries without rerunning the experiment."""

import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("edge-noise.csv"))
    parser.add_argument("--output", type=Path, default=Path("edge-noise.png"))
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open()))
    labels = [
        "Independent columns",
        "16-column blocks",
        "128-column blocks",
        "Common offset ±1 px",
    ]
    blocks = [1, 16, 128]
    blue, orange, purple = "#2563a6", "#d97720", "#8753a2"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (error, counts) = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.6),
        gridspec_kw={"width_ratios": [1.6, 1]},
        layout="constrained",
    )
    for sigma, color, offset in [(1, blue, -0.12), (2, orange, 0.12)]:
        for index, block in enumerate(blocks):
            selected = [
                r
                for r in rows
                if r["scenario"] == "noise"
                and int(r["sigma_px"]) == sigma
                and int(r["block_px"]) == block
            ]
            values = [float(r["max_abs_tilt_error_deg"]) for r in selected]
            center = median(values)
            error.errorbar(
                center,
                index + offset,
                xerr=[[center - min(values)], [max(values) - center]],
                fmt="o",
                color=color,
                capsize=5,
                linewidth=2,
                label=f"Noise σ = {sigma} image px" if index == 0 else None,
            )
    bias = [r for r in rows if r["scenario"] == "common_bias"]
    values = [float(r["max_abs_tilt_error_deg"]) for r in bias]
    center = median(values)
    error.errorbar(
        center,
        3,
        xerr=[[center - min(values)], [max(values) - center]],
        fmt="D",
        color=purple,
        capsize=5,
        linewidth=2,
        label="Common ±1 px shift",
    )
    error.axvline(0.009, color="#555555", linestyle="--", linewidth=1.2)
    error.text(
        0.009, -0.62, "0.009° comparison target", ha="left", va="bottom", fontsize=9
    )
    error.set(
        yticks=range(4),
        yticklabels=labels,
        ylim=(3.6, -0.8),
        xlim=(0, 0.09),
        xlabel="Worst absolute tilt error (°)",
        title="Error grows with spatial correlation",
    )
    error.legend(loc="upper right", frameon=False, fontsize=9)

    for index, block in enumerate([*blocks, None]):
        selected = (
            [
                r
                for r in rows
                if r["scenario"] == "noise"
                and r["sigma_px"] == "1"
                and int(r["block_px"]) == block
            ]
            if block
            else bias
        )
        within = sum(int(r["within_0p009_deg"]) for r in selected)
        total = sum(int(r["trials"]) for r in selected)
        counts.barh(index, within, color=blue if block else purple, height=0.5)
        counts.text(within + 5, index, f"{within}/{total}", va="center", fontsize=10)
    counts.set(
        yticks=range(4),
        yticklabels=[],
        ylim=(3.6, -0.8),
        xlim=(0, 310),
        xticks=[0, 60, 120, 180, 240],
        xlabel="Trials within 0.009°",
        title="One-pixel errors: passing / tested",
    )
    for axis in (error, counts):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Skyline edge noise: correlation and bias", fontsize=15, fontweight="bold"
    )
    fig.savefig(args.output, dpi=180, facecolor="white")


if __name__ == "__main__":
    main()

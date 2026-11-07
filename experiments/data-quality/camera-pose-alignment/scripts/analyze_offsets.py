"""Summarise offsets: per-pose stats, day/night validity, calibration, plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from camera_pose_alignment.registration import DEG_PER_PX  # noqa: E402


def q(s, p):
    return float(np.nanpercentile(s, p)) if len(s) else np.nan


def load(path: Path, min_cc: float, ir_sat: float) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["t"])
    d["local"] = d.t.dt.tz_localize("UTC").dt.tz_convert("Europe/Paris")
    d["hour"] = d.local.dt.hour
    d["ir"] = d.sat < ir_sat  # IR / night images are grey
    d["valid"] = d.cc >= min_cc
    d["pose"] = d.camera_name + "/" + d.detection_pose_id.astype(str)
    v = d[d.valid]
    med = v.groupby("pose")[["dx", "dy"]].median().add_prefix("med_")
    d = d.join(med, on="pose")
    # offset relative to the per-pose median position (removes reference bias)
    d["rx"], d["ry"] = d.dx - d.med_dx, d.dy - d.med_dy
    d["r"] = np.hypot(d.rx, d.ry)
    d["m_ref"] = np.hypot(d.dx, d.dy)
    return d


def per_pose(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pose, g in d.groupby("pose"):
        v = g[g.valid]
        rows.append(dict(
            pose=pose, n=len(g), valid_pct=100 * g.valid.mean(),
            valid_day_pct=100 * g[~g.ir].valid.mean(),
            valid_ir_pct=100 * g[g.ir].valid.mean() if g.ir.any() else np.nan,
            ref_bias_px=float(np.hypot(g.med_dx.iloc[0], g.med_dy.iloc[0])),
            med_vs_ref_px=q(v.m_ref, 50), med_px=q(v.r, 50), p95_px=q(v.r, 95),
            max_px=float(v.r.max()) if len(v) else np.nan,
            gt1px_pct=100 * (v.r > 1).mean() if len(v) else np.nan,
            gt5px_pct=100 * (v.r > 5).mean() if len(v) else np.nan,
        ))  # fmt: skip
    t = pd.DataFrame(rows).set_index("pose")
    t["p95_deg"] = t.p95_px * DEG_PER_PX
    t["max_deg"] = t.max_px * DEG_PER_PX
    return t.round(2)


def plot_timelines(d: pd.DataFrame, out: Path, ylim: float):
    poses = sorted(d.pose.unique())
    nc = 4
    nr = -(-len(poses) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(22, 2.6 * nr), sharey=True)
    for ax, pose in zip(axes.flat, poses, strict=False):
        g = d[d.pose == pose]
        v = g[g.valid]
        ax.scatter(v.t, v.rx, s=3, label="dx")
        ax.scatter(v.t, v.ry, s=3, label="dy")
        ax.set_title(f"{pose}  ({len(v)}/{len(g)} valid)", fontsize=9)
        ax.axhline(0, c="k", lw=0.5)
        ax.set_ylim(-ylim, ylim)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(poses) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, markerscale=3)
    fig.supylabel("offset vs per-pose median (px)  —  1 px = 0.042°")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_validity(d: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    h = d.groupby("hour").agg(valid=("valid", "mean"), ir=("ir", "mean"))
    ax[0].bar(h.index, 100 * h.valid, label="valid (cc ≥ thr)")
    ax[0].plot(h.index, 100 * h.ir, "k.-", label="IR / grey image")
    ax[0].set(xlabel="local hour", ylabel="%", title="Validity by hour")
    ax[0].legend(fontsize=8)
    m = d.dropna(subset=["cc"])
    ax[1].scatter(m.cc, m.r.clip(upper=50), s=2, alpha=0.3)
    ax[1].set(xlabel="ECC correlation", ylabel="|offset| vs median (px, clip 50)",
              yscale="symlog", title="Magnitude vs correlation")  # fmt: skip
    bins = pd.cut(m.cc, np.arange(0, 1.01, 0.05))
    b = m.groupby(bins, observed=True).r.median()
    ax[2].plot([i.mid for i in b.index], b.values, "o-")
    ax[2].set(xlabel="ECC correlation bin", ylabel="median |offset| (px)",
              yscale="log", title="Median magnitude per correlation bin")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def calibration(syn: pd.DataFrame, nb: pd.DataFrame, min_cc: float, out: Path):
    syn = syn.assign(ex=syn.dx - syn.tx, ey=syn.dy - syn.ty)
    s = syn.groupby("tx").agg(bias_x=("ex", "median"), bias_y=("ey", "median"))
    nbv = nb[nb.cc >= min_cc]
    mag = np.hypot(nbv.dx, nbv.dy)
    frac = (nbv.dx - nbv.dx.round()).abs()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(s.index, s.bias_x, "o-", label="median bias x")
    ax[0].plot(s.index, s.bias_y, "s-", label="median bias y")
    ax[0].axhline(0, c="k", lw=0.5)
    ax[0].set(xlabel="true shift x (px), y = x/2", ylabel="estimate - truth (px)",
              title="Synthetic sub-pixel shifts")  # fmt: skip
    ax[0].legend()
    ax[1].hist(frac, bins=20)
    ax[1].set(xlabel="|dx - round(dx)|  (0 = integer)", ylabel="count",
              title=f"Neighbour pairs: fractional part (n={len(nbv)})")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return dict(
        synthetic_max_abs_bias_px=float(s[["bias_x", "bias_y"]].abs().max().max()),
        synthetic_rmse_x_px=float(np.sqrt((syn.ex**2).mean())),
        neighbours_n=int(len(nb)), neighbours_valid=int(len(nbv)),
        neighbours_median_px=q(mag, 50), neighbours_p95_px=q(mag, 95),
        neighbours_max_px=float(mag.max()) if len(mag) else np.nan,
    )  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offsets", type=Path, required=True)
    p.add_argument("--calib-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-cc", type=float, default=0.5)
    p.add_argument("--ir-sat", type=float, default=12)
    p.add_argument("--ylim", type=float, default=10)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    d = load(a.offsets, a.min_cc, a.ir_sat)
    t = per_pose(d)
    t.to_csv(a.output_dir / "per_pose.csv")
    plot_timelines(d, a.output_dir / "timelines.png", a.ylim)
    plot_validity(d, a.output_dir / "validity.png")

    v = d[d.valid]
    both = d.valid & (d.pcc >= a.min_cc)
    seeded_vs_identity = np.hypot(d.dx - d.dx0, d.dy - d.dy0)[d.valid]
    metrics = dict(
        n_images=len(d), n_poses=int(d.pose.nunique()),
        valid_pct=100 * d.valid.mean(), valid_day_pct=100 * d[~d.ir].valid.mean(),
        valid_ir_pct=100 * d[d.ir].valid.mean(), ir_pct=100 * d.ir.mean(),
        median_vs_ref_px=q(v.m_ref, 50), median_vs_median_px=q(v.r, 50),
        p95_vs_median_px=q(v.r, 95), p99_vs_median_px=q(v.r, 99),
        gt1px_pct=100 * (v.r > 1).mean(), gt5px_pct=100 * (v.r > 5).mean(),
        consecutive_median_px=q(np.hypot(d.pdx, d.pdy)[both], 50),
        consecutive_p95_px=q(np.hypot(d.pdx, d.pdy)[both], 95),
        seeded_vs_identity_gt1px=int((seeded_vs_identity > 1).sum()),
        valid_identity_init_pct=100 * (d.cc0 >= a.min_cc).mean(),
        deg_per_px=DEG_PER_PX,
    )  # fmt: skip
    syn = pd.read_csv(a.calib_dir / "calib_synthetic.csv")
    nb = pd.read_csv(a.calib_dir / "calib_neighbours.csv")
    cal_png = a.output_dir / "calibration.png"
    metrics["calibration"] = calibration(syn, nb, a.min_cc, cal_png)
    js = json.dumps(metrics, indent=2, default=float)
    (a.output_dir / "metrics.json").write_text(js)
    print(js)
    print(t.to_string())
    big = v[v.r > 5].sort_values("t")[["pose", "local", "dx", "dy", "r", "cc", "ir"]]
    big.to_csv(a.output_dir / "large_offsets.csv", index=False)
    print(f"\n{len(big)} valid images > 5 px from pose median")


if __name__ == "__main__":
    main()

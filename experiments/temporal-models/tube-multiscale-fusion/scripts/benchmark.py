"""Benchmark a trained tube-multiscale-fusion checkpoint.

Computes the standardized Pyronear R&D metrics (see experiments/GUIDELINES.md):

  * Recall @ FPR     -- tube-level recall at fixed false-positive rates
  * Time-to-detection -- frames/seconds from tube start to first positive
                         classification, via prefix masking
  * Inference latency -- milliseconds per sequence and per frame (GPU + CPU)
  * Model size        -- trainable / total parameter count and FLOPs

Writes a JSON report and prints a human-readable summary.

Example:
    uv run python scripts/benchmark.py \\
        --data-dir data/05_model_input/val \\
        --checkpoint data/06_models/dinov2_multiscale/best_checkpoint.pt \\
        --predictions data/08_reporting/dinov2_multiscale/val/predictions.json \\
        --output data/08_reporting/dinov2_multiscale/val/benchmark.json
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.flop_counter import FlopCounterMode

from tube_multiscale_fusion.dataset import TubePatchDataset
from tube_multiscale_fusion.lit_module import LitTubeMultiscaleClassifier

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")


def _parse_ts(frame_id: str) -> datetime | None:
    m = _TS_RE.search(frame_id)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


def recall_at_fpr(predictions: list[dict], fprs: list[float]) -> dict[str, float]:
    """Tube-level recall at fixed false-positive rates from saved probabilities."""
    labels = np.array([p["truth"] for p in predictions])
    probs = np.array([p["prob"] for p in predictions])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {f"recall@fpr={f}": float("nan") for f in fprs}
    fpr, tpr, _ = roc_curve(labels, probs)
    out: dict[str, float] = {}
    for target in fprs:
        # Highest TPR achievable without exceeding the target FPR.
        feasible = tpr[fpr <= target]
        out[f"recall@fpr={target}"] = float(feasible.max()) if len(feasible) else 0.0
    return out


def median_frame_dt_seconds(ds: TubePatchDataset) -> float:
    """Median inter-frame interval across all sequences, from frame-id timestamps."""
    deltas: list[float] = []
    for rec in ds.index:
        meta = json.loads((ds.split_dir / rec["sequence_id"] / "meta.json").read_text())
        ts = [_parse_ts(f["frame_id"]) for f in meta["frames"]]
        ts = [t for t in ts if t is not None]
        for a, b in zip(ts, ts[1:], strict=False):
            deltas.append((b - a).total_seconds())
    return float(np.median(deltas)) if deltas else float("nan")


@torch.no_grad()
def time_to_detection(
    lit: LitTubeMultiscaleClassifier,
    ds: TubePatchDataset,
    device: torch.device,
    dt_seconds: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """For each positive tube, smallest prefix length whose prob >= threshold.

    Feeds increasing prefixes (k = 2..T frames) by masking later frames, exactly
    the variable-length contract the model was trained with. The reported time is
    ``(k_fire - 1) * dt`` -- elapsed wall-clock from the first tube frame to the
    frame at which the classifier first crosses the threshold.
    """
    lit.eval()
    first_fire_frames: list[int] = []
    never_fired = 0
    n_pos = 0
    for i in range(len(ds)):
        item = ds[i]
        if int(item["label"].item()) != 1:
            continue
        n_pos += 1
        patches = item["patches"].unsqueeze(0).to(device)
        base_mask = item["mask"]
        n_valid = int(base_mask.sum().item())
        fired_at = None
        for k in range(2, n_valid + 1):
            mask_k = torch.zeros_like(base_mask)
            mask_k[:k] = True
            prob = torch.sigmoid(
                lit(patches, mask_k.unsqueeze(0).to(device))
            ).item()
            if prob >= threshold:
                fired_at = k
                break
        if fired_at is None:
            never_fired += 1
        else:
            first_fire_frames.append(fired_at)
    frames = np.array(first_fire_frames) if first_fire_frames else np.array([np.nan])
    return {
        "n_positive": n_pos,
        "n_detected": len(first_fire_frames),
        "n_never_fired": never_fired,
        "median_frames_to_detect": float(np.median(frames)),
        "mean_frames_to_detect": float(np.mean(frames)),
        "median_seconds_to_detect": float(np.median(frames) - 1) * dt_seconds,
        "mean_seconds_to_detect": float(np.mean(frames) - 1) * dt_seconds,
        "frame_dt_seconds": dt_seconds,
    }


def model_size(lit: LitTubeMultiscaleClassifier, device: torch.device) -> dict:
    total = sum(p.numel() for p in lit.model.parameters())
    trainable = sum(p.numel() for p in lit.model.parameters() if p.requires_grad)
    x = torch.randn(1, 16, 3, 224, 224, device=device)
    mask = torch.ones(1, 16, dtype=torch.bool, device=device)
    lit.eval()
    with FlopCounterMode(display=False) as fcm:
        lit(x, mask)
    total_flops = fcm.get_total_flops()
    return {
        "total_params": int(total),
        "total_params_millions": round(total / 1e6, 2),
        "trainable_params": int(trainable),
        "trainable_params_millions": round(trainable / 1e6, 2),
        "frozen_params_millions": round((total - trainable) / 1e6, 2),
        "flops_per_sequence": int(total_flops),
        "gflops_per_sequence": round(total_flops / 1e9, 1),
        "gflops_per_frame": round(total_flops / 1e9 / 16, 2),
    }


@torch.no_grad()
def latency(
    lit: LitTubeMultiscaleClassifier,
    device: torch.device,
    n_warmup: int = 5,
    n_iters: int = 30,
) -> dict[str, float]:
    lit.eval()
    x = torch.randn(1, 16, 3, 224, 224, device=device)
    mask = torch.ones(1, 16, dtype=torch.bool, device=device)
    for _ in range(n_warmup):
        lit(x, mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        lit(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "ms_per_sequence_mean": float(arr.mean()),
        "ms_per_sequence_std": float(arr.std()),
        "ms_per_frame_mean": float(arr.mean() / 16),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    ds = TubePatchDataset(args.data_dir, max_frames=args.max_frames)

    lit = LitTubeMultiscaleClassifier.load_from_checkpoint(
        str(args.checkpoint), pretrained=False
    )
    lit.eval()

    report: dict = {"hardware": {}}
    gpu = torch.cuda.is_available()
    report["hardware"]["gpu"] = torch.cuda.get_device_name(0) if gpu else None
    report["hardware"]["cpu_threads"] = torch.get_num_threads()

    print("Computing recall @ FPR ...")
    report["recall_at_fpr"] = recall_at_fpr(predictions, [0.01, 0.05, 0.1])

    print("Computing model size + FLOPs ...")
    size_device = torch.device("cuda" if gpu else "cpu")
    lit.to(size_device)
    report["model_size"] = model_size(lit, size_device)

    print("Computing time-to-detection (prefix masking) ...")
    dt = median_frame_dt_seconds(ds)
    report["time_to_detection"] = time_to_detection(lit, ds, size_device, dt)

    print("Measuring GPU latency ..." if gpu else "Skipping GPU latency (no CUDA)")
    if gpu:
        lit.to(torch.device("cuda"))
        report["latency_gpu"] = latency(lit, torch.device("cuda"))

    print("Measuring CPU latency ...")
    lit.to(torch.device("cpu"))
    report["latency_cpu"] = latency(lit, torch.device("cpu"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Package a trained tube-multiscale-fusion checkpoint into a deployable .zip.

Builds a leaderboard-comparable model archive (mirroring the sibling
``bbox-tube-temporal`` experiment): bundles the YOLO companion detector
weights, the Lightning checkpoint, and the inference config needed by
``TubeMultiscaleFusionModel`` to run the full raw-frames -> decision pipeline.

The decision threshold is calibrated on the val patches to hit
``package.target_recall`` before being pinned into the archive config.

Example:
    uv run python scripts/package_model.py \\
        --checkpoint data/06_models/dinov2_multiscale/best_checkpoint.pt \\
        --params-path params.yaml \\
        --params-key train_dinov2_multiscale \\
        --yolo-weights-path data/01_raw/models/best.pt \\
        --val-patches-dir data/05_model_input/val \\
        --output data/07_model_output/dinov2_multiscale/model_package.zip
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tube_multiscale_fusion.dataset import IMAGENET_MEAN, IMAGENET_STD, TubePatchDataset
from tube_multiscale_fusion.package import _load_classifier, build_model_package


def _calibrate_threshold(
    probs: np.ndarray, labels: np.ndarray, *, target_recall: float
) -> float:
    """Smallest prob threshold whose recall on positives >= ``target_recall``.

    Same rule as ``bbox_tube_temporal``'s packager (that helper lives in the
    experiment, not the shared lib, so it's inlined here).
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall!r}")
    pos_probs = np.sort(probs[labels == 1])
    if pos_probs.size == 0:
        raise ValueError("no positives in labels; cannot calibrate recall")
    n_drop = int(np.floor(pos_probs.size * (1.0 - target_recall)))
    return float(pos_probs[n_drop])


def _collect_val_probabilities(
    classifier: torch.nn.Module,
    val_patches_dir: Path,
    *,
    max_frames: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the classifier over val patches; return ``(probs, labels)`` arrays."""
    ds = TubePatchDataset(val_patches_dir, max_frames=max_frames)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    probs: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = classifier(batch["patches"].to(device), batch["mask"].to(device))
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["label"].tolist())
    return np.asarray(probs), np.asarray(labels)


def _build_config(
    all_params: dict,
    package_params: dict,
    *,
    max_frames: int,
    threshold: float,
) -> dict:
    """Assemble the inference config embedded in the archive."""
    return {
        "infer": package_params["infer"],
        "tubes": {
            "iou_threshold": all_params["tubes"]["iou_threshold"],
            "max_misses": all_params["tubes"]["max_misses"],
            "min_tube_length": all_params["build_tubes"]["min_tube_length"],
            "infer_min_tube_length": package_params["infer_min_tube_length"],
            "min_detected_entries": all_params["build_tubes"]["min_detected_entries"],
            "interpolate_gaps": True,
        },
        "model_input": {
            "context_factor": all_params["model_input"]["context_factor"],
            "patch_size": all_params["model_input"]["patch_size"],
            "normalization": {
                "mean": IMAGENET_MEAN.flatten().tolist(),
                "std": IMAGENET_STD.flatten().tolist(),
            },
        },
        "classifier": {"max_frames": max_frames},
        "decision": {
            "aggregation": "max_logit",
            "threshold": float(threshold),
            "target_recall": package_params["target_recall"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--yolo-weights-path",
        type=Path,
        default=Path("data/01_raw/models/best.pt"),
    )
    parser.add_argument(
        "--val-patches-dir",
        type=Path,
        default=Path("data/05_model_input/val"),
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    all_params = yaml.safe_load(args.params_path.read_text())
    if args.params_key not in all_params:
        raise KeyError(f"params key not found in {args.params_path}: {args.params_key}")
    if "package" not in all_params:
        raise KeyError(f"'package' section missing from {args.params_path}")
    train_cfg = all_params[args.params_key]
    package_params = all_params["package"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = _load_classifier(args.checkpoint).to(device)
    max_frames = int(classifier.hparams["max_frames"])

    probs, labels = _collect_val_probabilities(
        classifier,
        args.val_patches_dir,
        max_frames=max_frames,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 4),
        device=device,
    )
    threshold_prob = _calibrate_threshold(
        probs, labels, target_recall=package_params["target_recall"]
    )
    # calibrate_threshold returns a probability; the max_logit rule compares raw
    # logits, so convert via the inverse sigmoid (logit) once here. Clamp away
    # from 0/1 so a saturated sigmoid doesn't produce a +/-inf threshold.
    p = float(np.clip(threshold_prob, 1e-6, 1.0 - 1e-6))
    threshold_logit = float(np.log(p / (1.0 - p)))

    config = _build_config(
        all_params,
        package_params,
        max_frames=max_frames,
        threshold=threshold_logit,
    )
    build_model_package(
        yolo_weights_path=args.yolo_weights_path,
        classifier_ckpt_path=args.checkpoint,
        config=config,
        variant=args.params_key.removeprefix("train_"),
        output_path=args.output,
    )
    print(
        f"[package] wrote {args.output} "
        f"({args.output.stat().st_size / 1e6:.1f} MB) | "
        f"variant={args.params_key} target_recall={package_params['target_recall']} "
        f"prob_threshold={threshold_prob:.4f} logit_threshold={threshold_logit:.4f}"
    )


if __name__ == "__main__":
    main()

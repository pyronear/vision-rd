"""Build a deployable model archive for one smokeynet-adapted variant.

Usage:
    uv run python scripts/package_model.py \\
        --variant gru_convnext_finetune \\
        --output data/06_models/gru_convnext_finetune/model.zip
"""

import argparse
from pathlib import Path

import torch
import yaml

from bbox_tube_temporal.calibration import calibrate_threshold
from bbox_tube_temporal.package import build_model_package
from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier
from bbox_tube_temporal.val_predict import collect_val_probabilities


def _load_classifier_from_ckpt(
    ckpt_path: Path, variant_cfg: dict
) -> TemporalSmokeClassifier:
    model = TemporalSmokeClassifier(
        backbone=variant_cfg["backbone"],
        arch=variant_cfg["arch"],
        hidden_dim=variant_cfg["hidden_dim"],
        pretrained=False,
        num_layers=variant_cfg.get("num_layers", 1),
        bidirectional=variant_cfg.get("bidirectional", False),
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        raw = blob["state_dict"]
    else:
        raw = blob
    sd = {
        k.removeprefix("model."): v for k, v in raw.items() if k.startswith("model.")
    } or raw
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _build_config(
    all_params: dict, variant_cfg: dict, package_params: dict, threshold: float
) -> dict:
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
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "classifier": {
            "backbone": variant_cfg["backbone"],
            "arch": variant_cfg["arch"],
            "hidden_dim": variant_cfg["hidden_dim"],
            "num_layers": variant_cfg.get("num_layers", 1),
            "bidirectional": variant_cfg.get("bidirectional", False),
            "max_frames": variant_cfg["max_frames"],
            "pretrained": False,
        },
        "decision": {
            "aggregation": "max_logit",
            "threshold": float(threshold),
            "target_recall": package_params["target_recall"],
            "trigger_rule": "end_of_winner",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="e.g. gru_convnext_finetune")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Default: data/06_models/<variant>/best_checkpoint.pt",
    )
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

    all_params = yaml.safe_load(args.params_path.read_text())
    variant_key = f"train_{args.variant}"
    if variant_key not in all_params:
        raise KeyError(f"{variant_key} not found in {args.params_path}")
    variant_cfg = all_params[variant_key]

    if "package" not in all_params:
        raise KeyError(f"'package' section missing from {args.params_path}")
    package_params = all_params["package"]

    checkpoint = args.checkpoint_path or (
        Path("data/06_models") / args.variant / "best_checkpoint.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    classifier = _load_classifier_from_ckpt(checkpoint, variant_cfg)
    probs, labels = collect_val_probabilities(
        classifier,
        args.val_patches_dir,
        max_frames=variant_cfg["max_frames"],
        batch_size=variant_cfg.get("batch_size", 32),
        num_workers=variant_cfg.get("num_workers", 4),
    )
    threshold = calibrate_threshold(
        probs, labels, target_recall=package_params["target_recall"]
    )

    config = _build_config(all_params, variant_cfg, package_params, threshold)
    build_model_package(
        yolo_weights_path=args.yolo_weights_path,
        classifier_ckpt_path=checkpoint,
        config=config,
        variant=args.variant,
        output_path=args.output,
    )
    print(
        f"[package] wrote {args.output} | variant={args.variant} "
        f"threshold={threshold:.4f} target_recall={package_params['target_recall']}"
    )


if __name__ == "__main__":
    main()

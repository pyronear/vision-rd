"""Build a deployable model archive for one bbox-tube-temporal variant.

Usage:
    uv run python scripts/package_model.py \\
        --variant gru_convnext_finetune \\
        --output data/06_models/gru_convnext_finetune/model.zip
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from bbox_tube_temporal.calibration import calibrate_threshold
from bbox_tube_temporal.logistic_calibrator import (
    LogisticCalibrator,
    extract_features,
)
from bbox_tube_temporal.logistic_calibrator_fit import fit as fit_logistic_calibrator
from bbox_tube_temporal.model import BboxTubeTemporalModel
from bbox_tube_temporal.package import _load_yolo, build_model_package
from bbox_tube_temporal.package_predict import collect_pipeline_records
from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier
from bbox_tube_temporal.val_predict import collect_val_probabilities


def _classifier_kwargs(cfg: dict) -> dict:
    """Pick the subset of *cfg* that TemporalSmokeClassifier accepts.

    Works for both params.yaml variant blocks and packaged config["classifier"]
    dicts — each key falls back to the classifier's own default when absent.
    """
    kwargs: dict = {
        "backbone": cfg["backbone"],
        "arch": cfg["arch"],
        "hidden_dim": cfg["hidden_dim"],
        "pretrained": False,
        "num_layers": cfg.get("num_layers", 1),
        "bidirectional": cfg.get("bidirectional", False),
        "finetune": cfg.get("finetune", False),
        "finetune_last_n_blocks": cfg.get("finetune_last_n_blocks", 0),
        "max_frames": cfg.get("max_frames", 20),
        "global_pool": cfg.get("global_pool", "avg"),
    }
    for k in (
        "transformer_num_layers",
        "transformer_num_heads",
        "transformer_ffn_dim",
        "transformer_dropout",
        "img_size",
    ):
        if k in cfg:
            kwargs[k] = cfg[k]
    return kwargs


def _load_classifier_from_ckpt(
    ckpt_path: Path, variant_cfg: dict
) -> TemporalSmokeClassifier:
    model = TemporalSmokeClassifier(**_classifier_kwargs(variant_cfg))
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
    all_params: dict,
    variant_cfg: dict,
    package_params: dict,
    threshold: float,
    *,
    aggregation: str,
    logistic_threshold: float | None,
) -> dict:
    decision: dict = {
        "aggregation": aggregation,
        "threshold": float(threshold),
        "target_recall": package_params["target_recall"],
        "trigger_rule": "end_of_winner",
    }
    if logistic_threshold is not None:
        decision["logistic_threshold"] = float(logistic_threshold)

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
        "classifier": _classifier_kwargs(variant_cfg),
        "decision": decision,
    }


def _calibrated_probs(
    records: list[dict], calibrator: LogisticCalibrator
) -> np.ndarray:
    probs = []
    for r in records:
        kept = r["kept_tubes"]
        if not kept:
            probs.append(0.0)
        else:
            best = max(kept, key=lambda t: t["logit"])
            feats = extract_features(best, n_tubes=len(kept))
            probs.append(calibrator.predict_proba(feats))
    return np.array(probs)


def _labels_array(records: list[dict]) -> np.ndarray:
    return np.array([1 if r["label"] == "smoke" else 0 for r in records])


def _fit_logistic_calibrator_and_threshold(
    *,
    yolo_weights_path: Path,
    classifier: TemporalSmokeClassifier,
    pipeline_config: dict,
    raw_train_dir: Path,
    raw_val_dir: Path,
    target_recall: float,
) -> tuple[LogisticCalibrator, float]:
    """Run full-pipeline inference on train+val, fit calibrator, pick threshold.

    Package-time helper: produces the artifacts the logistic branch needs
    to embed in the model zip (the fitted :class:`LogisticCalibrator`
    plus a calibrated probability threshold at ``target_recall``).
    """
    yolo_model = _load_yolo(yolo_weights_path)
    fit_model = BboxTubeTemporalModel(
        yolo_model=yolo_model,
        classifier=classifier,
        config=pipeline_config,
    )

    train_records = collect_pipeline_records(model=fit_model, raw_dir=raw_train_dir)
    calibrator = fit_logistic_calibrator(train_records)
    print(
        f"[package] logistic calibrator fit on {len(train_records)} train "
        f"records; coefs={calibrator.coefficients.tolist()} "
        f"intercept={calibrator.intercept:.6f}"
    )

    val_records = collect_pipeline_records(model=fit_model, raw_dir=raw_val_dir)
    probs = _calibrated_probs(val_records, calibrator)
    labels = _labels_array(val_records)
    logistic_threshold = calibrate_threshold(probs, labels, target_recall=target_recall)
    return calibrator, float(logistic_threshold)


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
    parser.add_argument(
        "--raw-train-dir",
        type=Path,
        default=Path("data/01_raw/datasets/train"),
        help="Used only when variant aggregation is 'logistic'.",
    )
    parser.add_argument(
        "--raw-val-dir",
        type=Path,
        default=Path("data/01_raw/datasets/val"),
        help="Used only when variant aggregation is 'logistic'.",
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

    aggregation = package_params.get("aggregation", {}).get(args.variant, "max_logit")
    calibrator: LogisticCalibrator | None = None
    logistic_threshold: float | None = None
    if aggregation == "logistic":
        # Build a pipeline-only config so BboxTubeTemporalModel can run the
        # full inference pipeline during fitting; the decision branch uses
        # max_logit here because we re-decide manually via the fitted
        # calibrator. The calibrator + logistic threshold get embedded in
        # the final config below.
        pipeline_config = _build_config(
            all_params,
            variant_cfg,
            package_params,
            threshold,
            aggregation="max_logit",
            logistic_threshold=None,
        )
        calibrator, logistic_threshold = _fit_logistic_calibrator_and_threshold(
            yolo_weights_path=args.yolo_weights_path,
            classifier=classifier,
            pipeline_config=pipeline_config,
            raw_train_dir=args.raw_train_dir,
            raw_val_dir=args.raw_val_dir,
            target_recall=package_params["target_recall"],
        )

    config = _build_config(
        all_params,
        variant_cfg,
        package_params,
        threshold,
        aggregation=aggregation,
        logistic_threshold=logistic_threshold,
    )
    build_model_package(
        yolo_weights_path=args.yolo_weights_path,
        classifier_ckpt_path=checkpoint,
        config=config,
        variant=args.variant,
        output_path=args.output,
        calibrator=calibrator,
    )
    suffix = ""
    if aggregation == "logistic" and logistic_threshold is not None:
        suffix = f" logistic_threshold={logistic_threshold:.4f}"
    print(
        f"[package] wrote {args.output} | variant={args.variant} "
        f"aggregation={aggregation} threshold={threshold:.4f} "
        f"target_recall={package_params['target_recall']}{suffix}"
    )


if __name__ == "__main__":
    main()

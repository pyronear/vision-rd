"""Evaluate a global-branch ablation variant (no_spatial / weighted_mean).

Same metric/plot outputs as ``scripts/evaluate.py`` for direct comparability.
"""

import argparse
import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from tube_multiscale_fusion.dataset import TubePatchDataset
from tube_multiscale_fusion.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from tube_multiscale_fusion.lit_ablation import LitAblationGlobal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True)
    args = parser.parse_args()

    full_params = yaml.safe_load(args.params_path.read_text())
    cfg = full_params[args.params_key]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant = cfg["ablation_variant"]

    L.seed_everything(cfg["seed"], workers=True)

    lit = LitAblationGlobal.load_from_checkpoint(str(args.checkpoint), pretrained=False)
    lit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    lit.to(device)

    ds = TubePatchDataset(args.data_dir, max_frames=cfg["max_frames"])
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
    )

    all_probs: list[float] = []
    all_labels: list[float] = []
    all_sequence_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patches"].to(device)
            mask = batch["mask"].to(device)
            logits = lit(patches, mask)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch["label"].tolist())
            all_sequence_ids.extend(batch["sequence_id"])

    probs = np.asarray(all_probs)
    labels = np.asarray(all_labels)
    preds = (probs > 0.5).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    pr_auc = float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0
    roc_auc = (
        float(roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else 0.0
    )

    neg_total = tn + fp
    pos_total = tp + fn
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "confusion_matrix_normalized": {
            "fp_as_fp": tn / neg_total if neg_total > 0 else 0.0,
            "fp_as_smoke": fp / neg_total if neg_total > 0 else 0.0,
            "smoke_as_fp": fn / pos_total if pos_total > 0 else 0.0,
            "smoke_as_smoke": tp / pos_total if pos_total > 0 else 0.0,
        },
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    split_name = args.output_dir.parent.name
    cm_abs = np.array([[tn, fp], [fn, tp]], dtype=float)
    cm_norm = np.array(
        [
            [metrics["confusion_matrix_normalized"]["fp_as_fp"],
             metrics["confusion_matrix_normalized"]["fp_as_smoke"]],
            [metrics["confusion_matrix_normalized"]["smoke_as_fp"],
             metrics["confusion_matrix_normalized"]["smoke_as_smoke"]],
        ],
        dtype=float,
    )
    plot_confusion_matrix(
        cm_abs,
        args.output_dir / "confusion_matrix.png",
        title=f"ablation-{variant} / {split_name} (counts)",
        normalized=False,
    )
    plot_confusion_matrix(
        cm_norm,
        args.output_dir / "confusion_matrix_normalized.png",
        title=f"ablation-{variant} / {split_name} (row-normalized)",
        normalized=True,
    )
    plot_pr_curve(labels, probs, args.output_dir / "pr_curve.png", title="PR")
    plot_roc_curve(labels, probs, args.output_dir / "roc_curve.png", title="ROC")

    predictions = [
        {
            "sequence_id": seq_id,
            "truth": int(truth),
            "prob": float(prob),
            "predicted": int(pred),
            "correct": bool(int(truth) == int(pred)),
        }
        for seq_id, truth, prob, pred in zip(
            all_sequence_ids, labels, probs, preds, strict=True
        )
    ]
    predictions.sort(key=lambda r: r["prob"], reverse=True)
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2)
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

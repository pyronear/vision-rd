"""Evaluate a trained temporal classifier on a split.

Loads the best checkpoint, runs inference over the dataset, computes
classification metrics + PR/ROC curves, and writes them under
``--output-dir``.
"""

import argparse
import json
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from smokeynet_adapted.dataset import TubePatchDataset
from smokeynet_adapted.lit_temporal import LitTemporalClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["mean_pool", "gru"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.params_path.read_text())[args.params_key]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(cfg["seed"], workers=True)

    lit = LitTemporalClassifier.load_from_checkpoint(
        str(args.checkpoint),
        backbone=cfg["backbone"],
        arch=cfg["arch"],
        hidden_dim=cfg["hidden_dim"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        pretrained=False,
        num_layers=cfg.get("num_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
    )
    lit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    lit.to(device)

    ds = TubePatchDataset(args.data_dir, max_frames=cfg["max_frames"])
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    all_probs: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patches"].to(device)
            mask = batch["mask"].to(device)
            logits = lit(patches, mask)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].tolist())

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

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    p, r, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r, p)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR (AP={pr_auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(args.output_dir / "pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC (AUC={roc_auc:.3f})")
    fig.savefig(args.output_dir / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

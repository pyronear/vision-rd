"""Evaluate a trained temporal classifier on a split.

Loads the best checkpoint, runs inference over the dataset, computes
classification metrics + PR/ROC curves, and writes them under
``--output-dir``.
"""

import argparse
import json
import shutil
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


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    labels = ["fp", "smoke"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    vmax = float(matrix.max()) if matrix.size else 0.0
    threshold = vmax * 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value * 100:.1f}%" if normalized else f"{int(value)}"
            color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["mean_pool", "gru"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True)
    parser.add_argument("--render-tubes-dir", type=Path, required=True)
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
    all_sequence_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patches"].to(device)
            mask = batch["mask"].to(device)
            logits = lit(patches, mask)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
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
    fp_as_fp = tn / neg_total if neg_total > 0 else 0.0
    fp_as_smoke = fp / neg_total if neg_total > 0 else 0.0
    smoke_as_fp = fn / pos_total if pos_total > 0 else 0.0
    smoke_as_smoke = tp / pos_total if pos_total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "confusion_matrix_normalized": {
            "fp_as_fp": fp_as_fp,
            "fp_as_smoke": fp_as_smoke,
            "smoke_as_fp": smoke_as_fp,
            "smoke_as_smoke": smoke_as_smoke,
        },
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    split_name = args.output_dir.parent.name
    cm_abs = np.array([[tn, fp], [fn, tp]], dtype=float)
    cm_norm = np.array(
        [[fp_as_fp, fp_as_smoke], [smoke_as_fp, smoke_as_smoke]], dtype=float
    )
    plot_confusion_matrix(
        cm_abs,
        args.output_dir / "confusion_matrix.png",
        title=f"{args.arch} / {split_name} (counts)",
        normalized=False,
    )
    plot_confusion_matrix(
        cm_norm,
        args.output_dir / "confusion_matrix_normalized.png",
        title=f"{args.arch} / {split_name} (row-normalized)",
        normalized=True,
    )

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

    arch_name = args.output_dir.name
    split = args.output_dir.parent.name

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
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))

    errors_dir = args.output_dir / "errors"
    shutil.rmtree(errors_dir, ignore_errors=True)
    (errors_dir / "fp").mkdir(parents=True, exist_ok=True)
    (errors_dir / "fn").mkdir(parents=True, exist_ok=True)

    label_name = {0: "fp", 1: "smoke"}
    n_fp_copied = 0
    n_fn_copied = 0
    n_missing = 0
    for entry in predictions:
        if entry["correct"]:
            continue
        truth = entry["truth"]
        predicted = entry["predicted"]
        seq_id = entry["sequence_id"]
        prob = entry["prob"]
        truth_label = label_name[truth]
        src = args.render_tubes_dir / truth_label / f"{seq_id}.png"
        if truth == 0 and predicted == 1:
            bucket = "fp"
        elif truth == 1 and predicted == 0:
            bucket = "fn"
        else:
            continue
        dst = errors_dir / bucket / f"prob={prob:.3f}_{seq_id}.png"
        if not src.exists():
            print(f"WARNING: missing render {src}")
            n_missing += 1
            continue
        shutil.copy2(src, dst)
        if bucket == "fp":
            n_fp_copied += 1
        else:
            n_fn_copied += 1

    print(
        f"[{arch_name}/{split}] {n_fp_copied} FPs copied, "
        f"{n_fn_copied} FNs copied, {n_missing} renders missing"
    )


if __name__ == "__main__":
    main()

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
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from bbox_tube_temporal.dataset import TubePatchDataset
from bbox_tube_temporal.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from bbox_tube_temporal.lit_temporal import LitTemporalClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", choices=["mean_pool", "gru", "transformer"], required=True
    )
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
        transformer_num_layers=cfg.get("transformer_num_layers", 2),
        transformer_num_heads=cfg.get("transformer_num_heads", 6),
        transformer_ffn_dim=cfg.get("transformer_ffn_dim", 1536),
        transformer_dropout=cfg.get("transformer_dropout", 0.1),
        max_frames=cfg.get("max_frames", 20),
        global_pool=cfg.get("global_pool", "avg"),
        img_size=cfg.get("img_size"),
        use_cosine_warmup=cfg.get("use_cosine_warmup", False),
        warmup_frac=cfg.get("warmup_frac", 0.05),
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

    plot_pr_curve(
        labels,
        probs,
        args.output_dir / "pr_curve.png",
        title="PR",
    )

    plot_roc_curve(
        labels,
        probs,
        args.output_dir / "roc_curve.png",
        title="ROC",
    )

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

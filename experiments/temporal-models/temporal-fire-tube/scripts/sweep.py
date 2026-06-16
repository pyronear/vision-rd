"""Grid-search fire-tube parameters on cached inference results.

Sweeps over tube construction params and classifier hyperparams.  For each
combination, tubes are built, features extracted, a RF trained, and
sequence-level metrics computed.  Results are written to a CSV sorted by
F1 descending.

Usage:
    uv run python scripts/sweep.py \
        --infer-dir data/03_primary/train \
        --data-dir data/01_raw/datasets/train \
        --output-dir data/08_reporting/sweep/train/all \
        --filter-prefix pyronear
"""

import argparse
import csv
import itertools
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.classifier import predict_tubes, train_classifier
from src.data import is_wf_sequence
from src.detector import load_inference_results
from src.features import extract_tabular_features
from src.tube import build_tubes_for_sequence
from src.types import FrameResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Sweep grids
CONFIDENCE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4]
MAX_DETECTION_AREAS: list[float | None] = [None, 0.05, 0.10]
N_ESTIMATORS_VALUES = [60, 120]
MAX_DEPTH_VALUES = [10, 20]


def _compute_metrics(results: list[dict]) -> dict:
    """Quick metrics computation for sweep (avoids importing evaluator)."""
    tp = sum(1 for r in results if r["gt"] and r["pred"])
    fp = sum(1 for r in results if not r["gt"] and r["pred"])
    fn = sum(1 for r in results if r["gt"] and not r["pred"])
    tn = sum(1 for r in results if not r["gt"] and not r["pred"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    n_neg = fp + tn
    fpr = fp / n_neg if n_neg > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep fire-tube parameters.")
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filter-prefix", type=str, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all inference results
    infer_files = sorted(args.infer_dir.glob("*.json"))
    logger.info("Loading %d inference files...", len(infer_files))

    all_data: list[tuple[str, bool, list[FrameResult]]] = []
    for infer_path in infer_files:
        seq_id = infer_path.stem
        if args.filter_prefix and not seq_id.startswith(args.filter_prefix):
            continue
        frames = load_inference_results(infer_path)
        gt = is_wf_sequence(args.data_dir / seq_id)
        all_data.append((seq_id, gt, frames))
    logger.info("Loaded %d sequences.", len(all_data))

    # Build parameter grid
    combos = list(
        itertools.product(
            CONFIDENCE_THRESHOLDS,
            IOU_THRESHOLDS,
            MAX_DETECTION_AREAS,
            N_ESTIMATORS_VALUES,
            MAX_DEPTH_VALUES,
        )
    )
    logger.info("Running %d parameter combinations...", len(combos))

    rows = []
    for combo in tqdm(combos, desc="Sweep"):
        conf_thresh, iou_thresh, max_det_area, n_est, max_depth = combo

        # Build tubes and extract features for all sequences
        all_features = []
        all_labels = []
        seq_tube_counts = []  # (seq_idx, n_tubes)

        for seq_idx, (seq_id, gt, frames) in enumerate(all_data):
            image_dir = args.data_dir / seq_id / "images"
            tubes = build_tubes_for_sequence(
                frame_results=frames,
                image_dir=image_dir,
                sequence_id=seq_id,
                crop_size=64,
                max_tube_length=50,
                confidence_threshold=conf_thresh,
                max_detection_area=max_det_area,
                iou_threshold=iou_thresh,
            )
            for tube in tubes:
                tube.label = gt
                feat = extract_tabular_features(tube)
                all_features.append(feat)
                all_labels.append(int(gt))
            seq_tube_counts.append((seq_idx, len(tubes)))

        if not all_features:
            continue

        X = np.array(all_features)
        y = np.array(all_labels)

        # Skip if only one class
        if len(set(y)) < 2:
            continue

        # Train classifier
        clf = train_classifier(X, y, n_estimators=n_est, max_depth=max_depth)

        # Predict at sequence level
        results = []
        feat_idx = 0
        for seq_idx, (_seq_id, gt, _frames) in enumerate(all_data):
            n_tubes = seq_tube_counts[seq_idx][1]
            if n_tubes > 0:
                seq_features = X[feat_idx : feat_idx + n_tubes]
                preds, _confs = predict_tubes(seq_features, clf)
                is_positive_pred = bool(preds.any())
                feat_idx += n_tubes
            else:
                is_positive_pred = False
            results.append({"gt": gt, "pred": is_positive_pred})

        metrics = _compute_metrics(results)
        rows.append(
            {
                "confidence_threshold": conf_thresh,
                "iou_threshold": iou_thresh,
                "max_detection_area": max_det_area,
                "n_estimators": n_est,
                "max_depth": max_depth,
                **metrics,
            }
        )

    if not rows:
        logger.warning("No valid results. Check data and parameters.")
        return

    # Sort by F1 descending
    rows.sort(key=lambda r: -r["f1"])

    # Write CSV
    csv_path = args.output_dir / "sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d results to %s", len(rows), csv_path)

    # Print top 10
    logger.info("Top 10 by F1:")
    for row in rows[:10]:
        logger.info(
            "  conf=%.2f iou=%.2f area=%s n_est=%d depth=%d | "
            "P=%.3f R=%.3f F1=%.3f FPR=%.3f",
            row["confidence_threshold"],
            row["iou_threshold"],
            row["max_detection_area"],
            row["n_estimators"],
            row["max_depth"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["fpr"],
        )


if __name__ == "__main__":
    main()

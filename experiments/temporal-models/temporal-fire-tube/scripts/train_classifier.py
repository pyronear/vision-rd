"""Train a Random Forest classifier on tube features.

Loads feature vectors and labels from the training split, trains the
classifier, and saves the model to disk.

Usage:
    uv run python scripts/train_classifier.py \
        --feature-dir data/05_model_input/train \
        --output-dir data/06_models \
        --n-estimators 120 \
        --max-depth 20
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from src.classifier import save_model, train_classifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RF classifier on tube features."
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all feature files
    feature_files = sorted(args.feature_dir.glob("*.npz"))
    logger.info("Loading features from %d files.", len(feature_files))

    all_features = []
    all_labels = []
    for fpath in feature_files:
        data = np.load(fpath)
        if data["features"].shape[0] > 0:
            all_features.append(data["features"])
            all_labels.append(data["labels"])

    if not all_features:
        logger.warning("No training samples found. Skipping.")
        return

    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)

    logger.info(
        "Training on %d tubes (%d positive, %d negative).",
        len(y),
        int(y.sum()),
        int(len(y) - y.sum()),
    )

    clf = train_classifier(
        X,
        y,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    # Log training accuracy
    train_acc = clf.score(X, y)
    logger.info("Training accuracy: %.4f", train_acc)
    if hasattr(clf, "oob_score_"):
        logger.info("OOB score: %.4f", clf.oob_score_)

    output_path = args.output_dir / "classifier.pkl"
    save_model(clf, output_path)
    logger.info("Saved classifier to %s", output_path)


if __name__ == "__main__":
    main()

"""Random Forest classifier for fire-tube classification.

Trains a Random Forest on tabular features extracted from fire-tubes,
with class balancing to handle imbalanced datasets.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 120,
    max_depth: int = 20,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on tube features.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Label vector of shape ``(n_samples,)``, binary (0/1).
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted :class:`RandomForestClassifier`.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def predict_tubes(
    X: np.ndarray,
    clf: RandomForestClassifier,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and confidence scores for tube features.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        clf: Fitted classifier.

    Returns:
        Tuple of ``(predictions, confidences)`` where predictions is a
        boolean array and confidences is the probability of the positive class.
    """
    predictions = clf.predict(X).astype(bool)
    confidences = clf.predict_proba(X)[:, 1]
    return predictions, confidences


def save_model(clf: RandomForestClassifier, output_path: Path) -> None:
    """Save a trained classifier to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)


def load_model(model_path: Path) -> RandomForestClassifier:
    """Load a trained classifier from disk."""
    return joblib.load(model_path)

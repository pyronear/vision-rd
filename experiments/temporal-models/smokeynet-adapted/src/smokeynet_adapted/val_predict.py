"""Run a trained classifier over val patches and collect per-sample probabilities.

Used by ``scripts/package_model.py`` to calibrate the decision threshold
without touching the evaluate stage.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TubePatchDataset
from .temporal_classifier import TemporalSmokeClassifier


def collect_val_probabilities(
    classifier: TemporalSmokeClassifier,
    val_patches_dir: Path,
    *,
    max_frames: int,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the classifier over ``val_patches_dir`` and collect probs/labels.

    Args:
        classifier: A loaded (eval-mode) ``TemporalSmokeClassifier``.
        val_patches_dir: Directory with ``_index.json`` and per-sequence
            patch sub-dirs (output of ``build_model_input`` stage on val).
        max_frames: Same as training config.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        device: ``"cuda"`` / ``"cpu"``; auto-detect when ``None``.

    Returns:
        ``(probs, labels)`` as 1D numpy arrays.
    """
    if device is not None:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(dev).eval()

    ds = TubePatchDataset(val_patches_dir, max_frames=max_frames)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    probs: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patches"].to(dev)
            mask = batch["mask"].to(dev)
            logits = classifier(patches, mask)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["label"].tolist())

    return np.asarray(probs), np.asarray(labels)

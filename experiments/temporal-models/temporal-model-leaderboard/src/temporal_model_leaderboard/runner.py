"""Evaluate a single TemporalModel on the pyro-dataset test set."""

import logging
from pathlib import Path

from pyrocore import TemporalModel

from .dataset import get_sorted_frames, list_sequences
from .types import SequenceResult

logger = logging.getLogger(__name__)


def evaluate_model(
    model: TemporalModel,
    test_dir: Path,
) -> list[SequenceResult]:
    """Run a model on every test sequence and collect results.

    For each sequence:

    1. Discover sorted frame paths via :func:`get_sorted_frames`.
    2. Call ``model.load_sequence`` then ``model.predict`` to obtain both
       loaded frames (for TTD timestamp extraction) and the output.
    3. Compute time-to-detection for true positives.

    Sequences with no images are skipped with a warning.

    Args:
        model: A :class:`~pyrocore.TemporalModel` instance.
        test_dir: Path to the test split root (e.g.,
            ``.../sequential_test/test``).

    Returns:
        List of :class:`SequenceResult`, one per evaluated sequence.
    """
    sequences = list_sequences(test_dir)
    logger.info("Found %d sequences in %s", len(sequences), test_dir)

    results: list[SequenceResult] = []

    for seq_path, ground_truth in sequences:
        frame_paths = get_sorted_frames(seq_path)
        if not frame_paths:
            logger.warning("Skipping %s: no images found", seq_path.name)
            continue

        frames = model.load_sequence(frame_paths)
        output = model.predict(frames)

        ttd_seconds = _compute_ttd(
            ground_truth=ground_truth,
            predicted=output.is_positive,
            trigger_frame_index=output.trigger_frame_index,
            frames=frames,
        )

        results.append(
            SequenceResult(
                sequence_id=seq_path.name,
                ground_truth=ground_truth,
                predicted=output.is_positive,
                ttd_seconds=ttd_seconds,
            )
        )

    logger.info("Evaluated %d sequences", len(results))
    return results


def _compute_ttd(
    *,
    ground_truth: bool,
    predicted: bool,
    trigger_frame_index: int | None,
    frames: list,
) -> float | None:
    """Compute time-to-detection in seconds for a true positive.

    Returns ``None`` if the sequence is not a TP, if the trigger frame
    index is missing, or if timestamps are unavailable.
    """
    if not (ground_truth and predicted and trigger_frame_index is not None):
        return None

    first_ts = frames[0].timestamp if frames else None
    trigger_ts = (
        frames[trigger_frame_index].timestamp
        if trigger_frame_index < len(frames)
        else None
    )

    if first_ts is None or trigger_ts is None:
        return None

    return (trigger_ts - first_ts).total_seconds()

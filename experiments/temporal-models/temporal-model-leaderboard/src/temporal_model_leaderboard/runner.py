"""Evaluate a single TemporalModel on the pyro-dataset test set."""

import logging
from pathlib import Path

from pyrocore import TemporalModel
from tqdm import tqdm

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
    2. Call ``model.load_sequence`` then ``model.predict`` to obtain the
       model's decision and trigger frame index.
    3. Record TTD in frames (= ``trigger_frame_index``) for true positives.

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

    for seq_path, ground_truth in tqdm(sequences, desc="eval", unit="seq"):
        frame_paths = get_sorted_frames(seq_path)
        if not frame_paths:
            logger.warning("Skipping %s: no images found", seq_path.name)
            continue

        frames = model.load_sequence(frame_paths)
        output = model.predict(frames)

        ttd_frames = (
            output.trigger_frame_index
            if ground_truth
            and output.is_positive
            and output.trigger_frame_index is not None
            else None
        )

        results.append(
            SequenceResult(
                sequence_id=seq_path.name,
                ground_truth=ground_truth,
                predicted=output.is_positive,
                ttd_frames=ttd_frames,
            )
        )

    logger.info("Evaluated %d sequences", len(results))
    return results

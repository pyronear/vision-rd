"""Result types for leaderboard evaluation."""

from dataclasses import dataclass, field


@dataclass
class SequenceResult:
    """Evaluation result for a single sequence.

    Attributes:
        sequence_id: Unique identifier (typically the sequence directory name).
        ground_truth: ``True`` if wildfire (positive), ``False`` if false positive.
        predicted: The model's binary classification decision.
        ttd_frames: Time-to-detection in frames (0-based trigger index)
            for true positives, or ``None`` if not a TP or the model did
            not report a trigger frame.
    """

    sequence_id: str
    ground_truth: bool
    predicted: bool
    ttd_frames: int | None = None


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model on the test set.

    Attributes:
        model_name: Human-readable model identifier.
        num_sequences: Total number of sequences evaluated.
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        tn: True negatives.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
        fpr: FP / (FP + TN).
        mean_ttd_frames: Mean time-to-detection across TPs, or ``None``.
        median_ttd_frames: Median time-to-detection across TPs, or ``None``.
    """

    model_name: str
    num_sequences: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    mean_ttd_frames: float | None = None
    median_ttd_frames: float | None = None


@dataclass
class LeaderboardEntry:
    """One row of the leaderboard: a model with its metrics and per-sequence details."""

    metrics: ModelMetrics
    sequence_results: list[SequenceResult] = field(default_factory=list)

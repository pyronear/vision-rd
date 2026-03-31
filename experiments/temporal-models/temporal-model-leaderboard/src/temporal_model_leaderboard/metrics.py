"""Sequence-level classification metrics for leaderboard evaluation."""

import statistics

from .types import ModelMetrics, SequenceResult


def compute_metrics(
    model_name: str,
    results: list[SequenceResult],
) -> ModelMetrics:
    """Compute precision, recall, F1, FPR, and TTD from sequence results.

    Args:
        model_name: Human-readable model identifier.
        results: Per-sequence evaluation results.

    Returns:
        Aggregated :class:`ModelMetrics`.
    """
    tp = sum(1 for r in results if r.ground_truth and r.predicted)
    fp = sum(1 for r in results if not r.ground_truth and r.predicted)
    fn = sum(1 for r in results if r.ground_truth and not r.predicted)
    tn = sum(1 for r in results if not r.ground_truth and not r.predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    n_negative_gt = fp + tn
    fpr = fp / n_negative_gt if n_negative_gt > 0 else 0.0

    ttd_values = [
        r.ttd_seconds
        for r in results
        if r.ground_truth and r.predicted and r.ttd_seconds is not None
    ]
    mean_ttd = round(sum(ttd_values) / len(ttd_values), 1) if ttd_values else None
    median_ttd = round(statistics.median(ttd_values), 1) if ttd_values else None

    return ModelMetrics(
        model_name=model_name,
        num_sequences=len(results),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        fpr=round(fpr, 4),
        mean_ttd_seconds=mean_ttd,
        median_ttd_seconds=median_ttd,
    )

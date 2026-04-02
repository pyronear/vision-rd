"""LSTM-based temporal fusion for smoke tubes.

Each tube is processed independently as a temporal sequence.  Gap frames
(where YOLO missed a detection) are filled via linear interpolation of
the surrounding observed features.
"""

import torch
from torch import Tensor, nn

from .types import Tube


def interpolate_tube_features(
    features: dict[tuple[int, int], Tensor],
    tube: Tube,
) -> Tensor:
    """Build a dense feature sequence for a tube, interpolating gaps.

    Args:
        features: Mapping from ``(frame_idx, det_idx_within_frame)`` to
            a ``(d_model,)`` feature vector.  Only entries with actual
            detections are expected.
        tube: The tube whose entries define the temporal structure.

    Returns:
        ``(tube_length, d_model)`` tensor with interpolated gap features.
    """
    entries = tube.entries
    if not entries:
        raise ValueError(f"Tube {tube.tube_id} has no entries")

    # Collect observed (non-gap) features with their positions
    observed: list[tuple[int, Tensor]] = []
    for i, entry in enumerate(entries):
        if entry.detection is not None:
            key = (entry.frame_idx, _det_key(tube, i))
            if key in features:
                observed.append((i, features[key]))

    if not observed:
        raise ValueError(f"Tube {tube.tube_id} has no observed features")

    d_model = observed[0][1].shape[0]
    device = observed[0][1].device
    result = torch.zeros(len(entries), d_model, device=device)

    # Place observed features
    for pos, feat in observed:
        result[pos] = feat

    # Interpolate gaps
    for i, entry in enumerate(entries):
        if entry.detection is not None:
            continue

        # Find nearest observed before and after
        before = _nearest_before(observed, i)
        after = _nearest_after(observed, i)

        if before is not None and after is not None:
            b_pos, b_feat = before
            a_pos, a_feat = after
            t = (i - b_pos) / (a_pos - b_pos)
            result[i] = b_feat * (1 - t) + a_feat * t
        elif before is not None:
            result[i] = before[1]
        elif after is not None:
            result[i] = after[1]

    return result


def _det_key(tube: Tube, entry_idx: int) -> int:
    """Return a detection index key within the frame for tube lookup.

    For tube feature lookup, we use 0 as a placeholder since features
    are indexed per (frame_idx, detection_index_in_frame) in the
    precomputed data.  The caller must build the features dict
    accordingly.
    """
    return 0


def _nearest_before(
    observed: list[tuple[int, Tensor]], pos: int
) -> tuple[int, Tensor] | None:
    """Find the nearest observed feature before position ``pos``."""
    best = None
    for obs_pos, feat in observed:
        if obs_pos < pos:
            best = (obs_pos, feat)
    return best


def _nearest_after(
    observed: list[tuple[int, Tensor]], pos: int
) -> tuple[int, Tensor] | None:
    """Find the nearest observed feature after position ``pos``."""
    for obs_pos, feat in observed:
        if obs_pos > pos:
            return (obs_pos, feat)
    return None


class TemporalLSTM(nn.Module):
    """Forward-only LSTM for temporal fusion of tube features.

    Processes each tube independently.  Causal (unidirectional) to
    preserve meaningful time-to-detection and streaming compatibility.

    Args:
        d_model: Input and output feature dimension.
        num_layers: Number of LSTM layers.
        dropout: Dropout between LSTM layers (only used if num_layers > 1).
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, tube_features: list[Tensor]) -> Tensor:
        """Process a list of tube feature sequences.

        Args:
            tube_features: List of ``(tube_length_i, d_model)`` tensors,
                one per tube.

        Returns:
            ``(num_tubes, d_model)`` tensor: mean-pooled LSTM output per tube.
        """
        if not tube_features:
            return torch.zeros(0, self.d_model)

        outputs = []
        for tube_seq in tube_features:
            # (1, T, d_model)
            x = tube_seq.unsqueeze(0)
            out, _ = self.lstm(x)  # (1, T, d_model)
            # Mean pool over time
            pooled = out.squeeze(0).mean(dim=0)  # (d_model,)
            outputs.append(pooled)

        stacked = torch.stack(outputs, dim=0)  # (num_tubes, d_model)
        return self.output_proj(stacked)

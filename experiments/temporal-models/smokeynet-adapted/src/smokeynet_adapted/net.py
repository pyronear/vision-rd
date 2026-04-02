"""Full SmokeyNetAdapted model combining all components.

Architecture: RoI features -> LSTM (per tube) -> ViT (across tubes) -> heads.
"""

import torch
from torch import Tensor, nn

from .heads import DetectionClassificationHead, SequenceClassificationHead
from .spatial_attention import SpatialAttentionViT
from .temporal_fusion import TemporalLSTM, interpolate_tube_features
from .types import Tube


class SmokeyNetAdapted(nn.Module):
    """Full adapted SmokeyNet model.

    Composes temporal LSTM, spatial ViT attention, and classification heads.
    In training mode, receives precomputed RoI features.  In inference mode,
    the caller provides features extracted by :class:`YoloRoiExtractor`.

    Args:
        d_model: Feature dimension throughout the model.
        lstm_layers: Number of LSTM layers.
        spatial_layers: Number of ViT transformer layers.
        spatial_heads: Number of attention heads in ViT.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        lstm_layers: int = 2,
        spatial_layers: int = 4,
        spatial_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.temporal_lstm = TemporalLSTM(
            d_model=d_model,
            num_layers=lstm_layers,
            dropout=dropout,
        )
        self.spatial_vit = SpatialAttentionViT(
            d_model=d_model,
            nhead=spatial_heads,
            num_layers=spatial_layers,
            dropout=dropout,
        )
        self.detection_head = DetectionClassificationHead(d_model=d_model)
        self.sequence_head = SequenceClassificationHead(d_model=d_model)
        # Learnable default logit for sequences with no detections
        self.empty_logit = nn.Parameter(torch.tensor([-5.0]))

    def forward(
        self,
        roi_features: Tensor,
        frame_indices: Tensor,
        bbox_coords: Tensor,
        tubes: list[Tube],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Forward pass from precomputed RoI features.

        Args:
            roi_features: ``(total_dets, d_model)`` precomputed RoI features
                for all detections across all frames.
            frame_indices: ``(total_dets,)`` frame index for each detection.
            bbox_coords: ``(total_dets, 4)`` normalised bboxes
                ``(cx, cy, w, h)`` for each detection.
            tubes: List of :class:`Tube` objects defining temporal structure.

        Returns:
            Tuple of:
            - ``sequence_logit``: ``(1,)`` raw logit for sequence classification.
            - ``intermediate_logits``: dict with ``"cnn"`` key mapping to
              ``(total_dets, 1)`` per-detection logits.
        """
        # Per-detection intermediate head
        det_logits = self.detection_head(roi_features)

        if roi_features.shape[0] == 0 or not tubes:
            return self.empty_logit, {"cnn": det_logits}

        # Build per-detection feature lookup
        features_dict = self._build_features_dict(roi_features, frame_indices)

        # LSTM temporal fusion per tube
        tube_seqs = []
        for tube in tubes:
            tube_seq = interpolate_tube_features(features_dict, tube)
            tube_seqs.append(tube_seq)

        tube_embeddings = self.temporal_lstm(tube_seqs)

        # Mean bbox per tube for spatial position encoding
        tube_bboxes = self._mean_tube_bboxes(tubes, bbox_coords, frame_indices)

        # ViT spatial attention
        cls_embedding = self.spatial_vit(tube_embeddings, tube_bboxes)

        # Sequence classification
        sequence_logit = self.sequence_head(cls_embedding)

        return sequence_logit, {"cnn": det_logits}

    @staticmethod
    def _build_features_dict(
        roi_features: Tensor, frame_indices: Tensor
    ) -> dict[tuple[int, int], Tensor]:
        """Map ``(frame_idx, det_idx_within_frame)`` -> feature vector."""
        features_dict: dict[tuple[int, int], Tensor] = {}
        frame_counts: dict[int, int] = {}
        for i in range(roi_features.shape[0]):
            f_idx = int(frame_indices[i].item())
            det_idx = frame_counts.get(f_idx, 0)
            frame_counts[f_idx] = det_idx + 1
            features_dict[(f_idx, det_idx)] = roi_features[i]
        return features_dict

    @staticmethod
    def _mean_tube_bboxes(
        tubes: list[Tube],
        bbox_coords: Tensor,
        frame_indices: Tensor,
    ) -> Tensor:
        """Compute mean bbox for each tube for spatial position encoding."""
        device = bbox_coords.device
        tube_bboxes = []
        for tube in tubes:
            coords = []
            for entry in tube.entries:
                if entry.detection is not None:
                    det = entry.detection
                    coords.append(
                        torch.tensor(
                            [det.cx, det.cy, det.w, det.h],
                            device=device,
                        )
                    )
            if coords:
                tube_bboxes.append(torch.stack(coords).mean(dim=0))
            else:
                tube_bboxes.append(torch.zeros(4, device=device))
        return torch.stack(tube_bboxes)

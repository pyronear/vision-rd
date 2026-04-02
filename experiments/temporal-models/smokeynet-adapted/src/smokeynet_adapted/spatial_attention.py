"""ViT-style spatial attention across tube embeddings.

Applies a Transformer encoder with a learnable CLS token to aggregate
spatial information across all tubes in a sequence.  Tube positions are
encoded via a small MLP on their mean bounding-box coordinates.
"""

import torch
from torch import Tensor, nn


class SpatialAttentionViT(nn.Module):
    """ViT-style spatial attention with learnable CLS token.

    Args:
        d_model: Feature dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Spatial position encoding from bbox coordinates (cx, cy, w, h)
        self.bbox_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self, tube_embeddings: Tensor, bbox_coords: Tensor | None = None
    ) -> Tensor:
        """Aggregate tube embeddings into a single sequence embedding.

        Args:
            tube_embeddings: ``(num_tubes, d_model)`` from the temporal LSTM.
            bbox_coords: ``(num_tubes, 4)`` mean bbox ``(cx, cy, w, h)``
                for spatial position encoding.  If ``None``, no spatial
                position is added.

        Returns:
            ``(d_model,)`` sequence-level embedding from the CLS token.
        """
        if tube_embeddings.shape[0] == 0:
            return torch.zeros(self.d_model, device=tube_embeddings.device)

        # Add spatial position encoding
        tokens = tube_embeddings.unsqueeze(0)  # (1, N, d_model)
        if bbox_coords is not None:
            pos = self.bbox_proj(bbox_coords).unsqueeze(0)  # (1, N, d_model)
            tokens = tokens + pos

        # Prepend CLS token
        cls = self.cls_token.expand(1, -1, -1)  # (1, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (1, N+1, d_model)

        # Transformer encoder
        out = self.encoder(tokens)  # (1, N+1, d_model)

        # Return CLS token output
        return out[0, 0]  # (d_model,)

"""Classification heads for the adapted SmokeyNet model.

Two heads to start:
- Per-detection head for intermediate CNN supervision.
- Sequence-level head on the CLS token for final classification.
"""

from torch import Tensor, nn


class DetectionClassificationHead(nn.Module):
    """Per-detection binary classification head.

    Applied to RoI features for intermediate supervision of the backbone.

    Args:
        d_model: Input feature dimension.
    """

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(N, d_model)`` per-detection features.

        Returns:
            ``(N, 1)`` raw logits (before sigmoid).
        """
        return self.head(x)


class SequenceClassificationHead(nn.Module):
    """Sequence-level binary classification head.

    Three FC layers with ReLU, matching SmokeyNet's design.

    Args:
        d_model: Input feature dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, d_model: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(d_model,)`` sequence-level CLS embedding.

        Returns:
            ``(1,)`` raw logit (before sigmoid).
        """
        return self.head(x)

"""PyTorch Lightning training module for SmokeyNetAdapted."""

import lightning as L
import torch
from torch import Tensor

from .net import SmokeyNetAdapted
from .types import Tube


class SmokeyNetLightningModule(L.LightningModule):
    """Lightning wrapper for SmokeyNetAdapted training.

    Args:
        d_model: Feature dimension.
        lstm_layers: Number of LSTM layers.
        spatial_layers: Number of ViT layers.
        spatial_heads: Number of attention heads.
        learning_rate: Base learning rate.
        weight_decay: AdamW weight decay.
        sequence_loss_weight: Weight for sequence-level BCE loss.
        detection_loss_weight: Weight for per-detection BCE loss.
        sequence_pos_weight: Positive class weight for sequence BCE.
        detection_pos_weight: Positive class weight for detection BCE.
        warmup_epochs: Number of warmup epochs for cosine scheduler.
        total_epochs: Total training epochs (for cosine scheduler).
    """

    def __init__(
        self,
        d_model: int = 512,
        lstm_layers: int = 2,
        spatial_layers: int = 4,
        spatial_heads: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        sequence_loss_weight: float = 1.0,
        detection_loss_weight: float = 1.0,
        sequence_pos_weight: float = 5.0,
        detection_pos_weight: float = 40.0,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.net = SmokeyNetAdapted(
            d_model=d_model,
            lstm_layers=lstm_layers,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
        )
        self.seq_bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([sequence_pos_weight])
        )
        self.det_bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([detection_pos_weight])
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.sequence_loss_weight = sequence_loss_weight
        self.detection_loss_weight = detection_loss_weight

    def forward(
        self,
        roi_features: Tensor,
        frame_indices: Tensor,
        bbox_coords: Tensor,
        tubes: list[Tube],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.net(roi_features, frame_indices, bbox_coords, tubes)

    def _compute_loss(
        self,
        seq_logit: Tensor,
        det_logits: Tensor,
        seq_label: Tensor,
        det_labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute combined loss with sequence and detection components."""
        seq_loss = self.seq_bce(seq_logit, seq_label.unsqueeze(0).to(seq_logit.device))
        if det_logits.shape[0] > 0:
            det_loss = self.det_bce(
                det_logits.squeeze(-1),
                det_labels.to(det_logits.device),
            )
        else:
            det_loss = torch.tensor(0.0, device=seq_logit.device)

        total = (
            self.sequence_loss_weight * seq_loss + self.detection_loss_weight * det_loss
        )
        return total, seq_loss, det_loss

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        roi_features = batch["roi_features"].requires_grad_(True)
        seq_logit, intermediates = self.net(
            roi_features,
            batch["frame_indices"],
            batch["bbox_coords"],
            batch["tubes"],
        )
        total, seq_loss, det_loss = self._compute_loss(
            seq_logit,
            intermediates["cnn"],
            batch["sequence_label"],
            batch["detection_labels"],
        )
        self.log("train_loss", total, prog_bar=True)
        self.log("train_seq_loss", seq_loss)
        self.log("train_det_loss", det_loss)
        return total

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        seq_logit, intermediates = self.net(
            batch["roi_features"],
            batch["frame_indices"],
            batch["bbox_coords"],
            batch["tubes"],
        )
        total, seq_loss, det_loss = self._compute_loss(
            seq_logit,
            intermediates["cnn"],
            batch["sequence_label"],
            batch["detection_labels"],
        )
        pred = (torch.sigmoid(seq_logit) > 0.5).float().item()
        gt = batch["sequence_label"].item()

        self.log("val_loss", total, prog_bar=True)
        self.log("val_seq_loss", seq_loss)
        self.log("val_det_loss", det_loss)
        self.log("val_correct", float(pred == gt))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.total_epochs - self.warmup_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

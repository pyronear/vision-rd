"""PyTorch Lightning wrapper around TemporalSmokeClassifier."""

import lightning as L
import torch
from torch import Tensor

from .temporal_classifier import TemporalSmokeClassifier


class LitTemporalClassifier(L.LightningModule):
    """Lightning module: BCE loss, AdamW on head params only.

    Args:
        backbone: timm model name.
        arch: ``"mean_pool"`` or ``"gru"``.
        hidden_dim: head hidden width.
        learning_rate: AdamW lr.
        weight_decay: AdamW weight decay.
        pretrained: whether to load pretrained backbone weights.
        num_layers: GRU layers (ignored when arch != gru).
        bidirectional: GRU direction (ignored when arch != gru).
        finetune: if True, unfreeze backbone blocks and use two param groups.
        finetune_last_n_blocks: number of backbone blocks to unfreeze.
        backbone_lr: learning rate for backbone params (required when finetune=True).
    """

    def __init__(
        self,
        backbone: str,
        arch: str,
        hidden_dim: int,
        learning_rate: float,
        weight_decay: float,
        pretrained: bool = True,
        num_layers: int = 1,
        bidirectional: bool = False,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        backbone_lr: float | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TemporalSmokeClassifier(
            backbone=backbone,
            arch=arch,
            hidden_dim=hidden_dim,
            pretrained=pretrained,
            num_layers=num_layers,
            bidirectional=bidirectional,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.finetune = finetune
        self.backbone_lr = backbone_lr
        self._val_preds: list[float] = []
        self._val_labels: list[float] = []

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        return self.model(patches, mask)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        logits = self(batch["patches"], batch["mask"])
        loss = self.loss_fn(logits, batch["label"])
        self.log("train/loss", loss, prog_bar=True, batch_size=logits.shape[0])
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["patches"], batch["mask"])
        loss = self.loss_fn(logits, batch["label"])
        probs = torch.sigmoid(logits).detach().cpu()
        labels = batch["label"].detach().cpu()
        self._val_preds.extend(probs.tolist())
        self._val_labels.extend(labels.tolist())
        self.log("val/loss", loss, prog_bar=True, batch_size=logits.shape[0])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        probs = torch.tensor(self._val_preds)
        labels = torch.tensor(self._val_labels)
        preds = (probs > 0.5).float()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()
        acc = (tp + tn) / (tp + tn + fp + fn).clamp(min=1)
        prec = tp / (tp + fp).clamp(min=1)
        rec = tp / (tp + fn).clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
        self.log("val/accuracy", acc, prog_bar=True)
        self.log("val/precision", prec)
        self.log("val/recall", rec)
        self.log("val/f1", f1, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        if not self.finetune:
            head_params = [p for p in self.model.head.parameters() if p.requires_grad]
            return torch.optim.AdamW(
                head_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        if self.backbone_lr is None:
            raise ValueError("backbone_lr must be set when finetune=True")

        backbone_params = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            [
                {
                    "params": head_params,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": backbone_params,
                    "lr": self.backbone_lr,
                    "weight_decay": self.weight_decay,
                },
            ]
        )

"""Lightning wrapper for the local-only 3D ResNet ablation classifier."""

from __future__ import annotations

import math

import lightning as L
import torch
from torch import Tensor

from .local_resnet3d import AblationLocalResnet3DClassifier


class LitAblationLocalResnet3D(L.LightningModule):
    """BCE loss; AdamW with two param groups (3D ResNet vs head) when finetuning.

    Mirrors the metric logging of ``LitTubeMultiscaleClassifier`` so the two
    models are directly comparable on the same val curves.
    """

    def __init__(
        self,
        grid_size: int,
        cell_size: int,
        tube_length: int,
        temporal_stride: int,
        resnet_model: str,
        resnet_pretrained: bool,
        resnet_finetune: bool,
        resnet_finetune_last_n_blocks: int,
        resnet_clip_spatial_size: int,
        embed_dim: int | None,
        head_hidden_dim: int,
        head_dropout: float,
        learning_rate: float,
        weight_decay: float,
        backbone_lr: float | None = None,
        use_cosine_warmup: bool = False,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AblationLocalResnet3DClassifier(
            grid_size=grid_size,
            cell_size=cell_size,
            tube_length=tube_length,
            temporal_stride=temporal_stride,
            resnet_model=resnet_model,
            resnet_pretrained=resnet_pretrained,
            resnet_finetune=resnet_finetune,
            resnet_finetune_last_n_blocks=resnet_finetune_last_n_blocks,
            resnet_clip_spatial_size=resnet_clip_spatial_size,
            embed_dim=embed_dim,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.backbone_lr = backbone_lr
        self.finetune = resnet_finetune
        self.use_cosine_warmup = use_cosine_warmup
        self.warmup_frac = warmup_frac
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

    def _head_params(self) -> list[torch.nn.Parameter]:
        encoder_ids = {id(p) for p in self.model.encoder.parameters()}
        return [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in encoder_ids
        ]

    def configure_optimizers(self):
        head_params = self._head_params()
        if not self.finetune:
            optimizer = torch.optim.AdamW(
                head_params, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            if self.backbone_lr is None:
                raise ValueError("backbone_lr must be set when resnet_finetune=True")
            encoder_params = [
                p for p in self.model.encoder.parameters() if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": head_params,
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay,
                    },
                    {
                        "params": encoder_params,
                        "lr": self.backbone_lr,
                        "weight_decay": self.weight_decay,
                    },
                ]
            )

        if not self.use_cosine_warmup:
            return optimizer

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(total_steps * self.warmup_frac))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

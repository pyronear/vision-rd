"""Lightning wrapper for the no-temporal-module ablation classifier.

Mirrors ``LitTubeMultiscaleClassifier`` exactly (BCE, AdamW, cosine warmup,
F1 metric logging) so the two models are directly comparable. There is no
backbone to fine-tune here (the global DINOv2 branch is removed), so a single
parameter group is used.
"""

from __future__ import annotations

import math

import lightning as L
import torch
from torch import Tensor

from .ablation_classifier import (
    AblationNoSpatialClassifier,
    AblationNoTemporalClassifier,
    AblationWeightedMeanClassifier,
)


class LitAblationNoTemporal(L.LightningModule):
    def __init__(
        self,
        grid_size: int,
        cell_size: int,
        tube_length: int,
        temporal_stride: int,
        local_t_kernel: int,
        local_h_patch: int,
        local_w_patch: int,
        local_embed_dim: int,
        local_num_layers: int,
        local_num_heads: int,
        local_ffn_dim: int,
        local_dropout: float,
        d_fusion: int,
        fusion_num_layers: int,
        fusion_num_heads: int,
        fusion_ffn_dim: int,
        fusion_dropout: float,
        head_hidden_dim: int,
        head_dropout: float,
        query_dim: int,
        learning_rate: float,
        weight_decay: float,
        use_cosine_warmup: bool = False,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = AblationNoTemporalClassifier(
            grid_size=grid_size,
            cell_size=cell_size,
            tube_length=tube_length,
            temporal_stride=temporal_stride,
            local_t_kernel=local_t_kernel,
            local_h_patch=local_h_patch,
            local_w_patch=local_w_patch,
            local_embed_dim=local_embed_dim,
            local_num_layers=local_num_layers,
            local_num_heads=local_num_heads,
            local_ffn_dim=local_ffn_dim,
            local_dropout=local_dropout,
            d_fusion=d_fusion,
            fusion_num_layers=fusion_num_layers,
            fusion_num_heads=fusion_num_heads,
            fusion_ffn_dim=fusion_ffn_dim,
            fusion_dropout=fusion_dropout,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            query_dim=query_dim,
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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


class LitAblationGlobal(L.LightningModule):
    """Lightning wrapper for the global-branch ablations.

    Handles two variants that both retain the global DINOv2 branch (and thus a
    fine-tunable backbone with a separate LR):

    - ``no_spatial``:    :class:`AblationNoSpatialClassifier` (global only)
    - ``weighted_mean``: :class:`AblationWeightedMeanClassifier`
      (cross-attention fusion → weighted means)
    """

    def __init__(
        self,
        variant: str,
        backbone: str,
        max_frames: int,
        global_aggregator_num_layers: int,
        global_aggregator_num_heads: int,
        global_aggregator_ffn_dim: int,
        global_aggregator_dropout: float,
        head_hidden_dim: int,
        head_dropout: float,
        learning_rate: float,
        weight_decay: float,
        # local + fusion (only used by weighted_mean)
        grid_size: int = 2,
        cell_size: int = 112,
        tube_length: int = 4,
        temporal_stride: int = 2,
        local_t_kernel: int = 2,
        local_h_patch: int = 14,
        local_w_patch: int = 14,
        local_embed_dim: int = 384,
        local_num_layers: int = 2,
        local_num_heads: int = 6,
        local_ffn_dim: int = 1536,
        local_dropout: float = 0.1,
        d_fusion: int = 384,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        backbone_lr: float | None = None,
        aggregator_kind: str = "transformer",
        img_size: int = 224,
        pretrained: bool = True,
        use_cosine_warmup: bool = False,
        warmup_frac: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if variant == "no_spatial":
            self.model: torch.nn.Module = AblationNoSpatialClassifier(
                backbone=backbone,
                finetune=finetune,
                finetune_last_n_blocks=finetune_last_n_blocks,
                max_frames=max_frames,
                global_aggregator_num_layers=global_aggregator_num_layers,
                global_aggregator_num_heads=global_aggregator_num_heads,
                global_aggregator_ffn_dim=global_aggregator_ffn_dim,
                global_aggregator_dropout=global_aggregator_dropout,
                head_hidden_dim=head_hidden_dim,
                head_dropout=head_dropout,
                aggregator_kind=aggregator_kind,
                img_size=img_size,
                pretrained=pretrained,
            )
        elif variant == "weighted_mean":
            self.model = AblationWeightedMeanClassifier(
                backbone=backbone,
                finetune=finetune,
                finetune_last_n_blocks=finetune_last_n_blocks,
                max_frames=max_frames,
                global_aggregator_num_layers=global_aggregator_num_layers,
                global_aggregator_num_heads=global_aggregator_num_heads,
                global_aggregator_ffn_dim=global_aggregator_ffn_dim,
                global_aggregator_dropout=global_aggregator_dropout,
                grid_size=grid_size,
                cell_size=cell_size,
                tube_length=tube_length,
                temporal_stride=temporal_stride,
                local_t_kernel=local_t_kernel,
                local_h_patch=local_h_patch,
                local_w_patch=local_w_patch,
                local_embed_dim=local_embed_dim,
                local_num_layers=local_num_layers,
                local_num_heads=local_num_heads,
                local_ffn_dim=local_ffn_dim,
                local_dropout=local_dropout,
                d_fusion=d_fusion,
                head_hidden_dim=head_hidden_dim,
                head_dropout=head_dropout,
                img_size=img_size,
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"unknown variant {variant!r}")

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.finetune = finetune
        self.backbone_lr = backbone_lr
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
        self._val_preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
        self._val_labels.extend(batch["label"].detach().cpu().tolist())
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

    def _non_backbone_params(self) -> list[torch.nn.Parameter]:
        backbone_ids = {id(p) for p in self.model.global_branch.backbone.parameters()}
        return [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]

    def configure_optimizers(self):
        non_backbone = self._non_backbone_params()
        if not self.finetune:
            optimizer = torch.optim.AdamW(
                non_backbone, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            if self.backbone_lr is None:
                raise ValueError("backbone_lr must be set when finetune=True")
            backbone_params = [
                p
                for p in self.model.global_branch.backbone.parameters()
                if p.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": non_backbone,
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

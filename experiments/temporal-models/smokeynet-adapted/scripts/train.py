"""Train the SmokeyNetAdapted model.

Loads precomputed RoI features from data/03_primary/, trains with
PyTorch Lightning, and saves the best checkpoint.

Usage:
    uv run python scripts/train.py \
        --train-dir data/03_primary/train \
        --val-dir data/03_primary/val \
        --output-dir data/06_models \
        --params-path params.yaml
"""

import argparse
import logging
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from smokeynet_adapted.dataset import SmokeyNetDataset
from smokeynet_adapted.training import SmokeyNetLightningModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SmokeyNetAdapted model.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    args = parser.parse_args()

    params = yaml.safe_load(args.params_path.read_text())
    train_cfg = params["train"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(train_cfg["seed"])

    # Datasets
    train_ds = SmokeyNetDataset(args.train_dir)
    val_ds = SmokeyNetDataset(args.val_dir)
    logger.info(
        "Train: %d sequences, Val: %d sequences",
        len(train_ds),
        len(val_ds),
    )

    def _collate_single(batch):
        """Identity collate for batch_size=1 (avoids stacking Tube objects)."""
        return batch[0]

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_single,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_single,
    )

    # Model
    module = SmokeyNetLightningModule(
        d_model=train_cfg["d_model"],
        lstm_layers=train_cfg["lstm_layers"],
        spatial_layers=train_cfg["spatial_layers"],
        spatial_heads=train_cfg["spatial_heads"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        sequence_loss_weight=train_cfg["sequence_loss_weight"],
        detection_loss_weight=train_cfg["detection_loss_weight"],
        sequence_pos_weight=train_cfg["sequence_pos_weight"],
        detection_pos_weight=train_cfg["detection_pos_weight"],
        warmup_epochs=train_cfg["warmup_epochs"],
        total_epochs=train_cfg["epochs"],
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best_checkpoint",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    # Loggers
    tb_logger = TensorBoardLogger(save_dir=args.output_dir, name="tb_logs")
    csv_logger = CSVLogger(save_dir=args.output_dir, name="csv_logs")

    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator=accelerator,
        devices=1,
        accumulate_grad_batches=train_cfg["gradient_accumulation_steps"],
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    trainer.fit(module, train_loader, val_loader)

    logger.info(
        "Training complete. Best checkpoint: %s",
        checkpoint_cb.best_model_path,
    )


if __name__ == "__main__":
    main()

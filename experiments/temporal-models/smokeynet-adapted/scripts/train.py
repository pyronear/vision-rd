"""Train the basic temporal smoke classifier (mean_pool or gru arch).

Reads a single named section from ``params.yaml`` (e.g. ``train_gru``)
so each DVC stage owns its own params.
"""

import argparse
import sys
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from smokeynet_adapted.dataset import TubePatchDataset
from smokeynet_adapted.lit_temporal import LitTemporalClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["mean_pool", "gru"], required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True, help="Key in params.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.params_path.read_text())[args.params_key]
    if cfg["arch"] != args.arch:
        raise ValueError(
            f"--arch={args.arch} mismatches "
            f"params[{args.params_key}].arch={cfg['arch']}"
        )

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(
        f"CUDA available: {torch.cuda.is_available()} | "
        f"device count: {torch.cuda.device_count()} | "
        f"device: {device_name}",
        file=sys.stderr,
        flush=True,
    )

    L.seed_everything(cfg["seed"], workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TubePatchDataset(args.train_dir, max_frames=cfg["max_frames"])
    val_ds = TubePatchDataset(args.val_dir, max_frames=cfg["max_frames"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )

    lit = LitTemporalClassifier(
        backbone=cfg["backbone"],
        arch=cfg["arch"],
        hidden_dim=cfg["hidden_dim"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        pretrained=True,
        num_layers=cfg.get("num_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
        finetune=cfg.get("finetune", False),
        finetune_last_n_blocks=cfg.get("finetune_last_n_blocks", 0),
        backbone_lr=cfg.get("backbone_lr"),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="best",
            monitor="val/f1",
            mode="max",
            save_top_k=1,
            save_weights_only=False,
        ),
        EarlyStopping(
            monitor="val/f1", mode="max", patience=cfg["early_stop_patience"]
        ),
    ]
    loggers = [
        CSVLogger(save_dir=args.output_dir, name="csv_logs"),
        TensorBoardLogger(save_dir=args.output_dir, name="tb_logs"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic=True,
        accelerator="auto",
        devices=1,
    )
    print(
        f"Trainer accelerator flag: {trainer._accelerator_connector._accelerator_flag}",
        file=sys.stderr,
        flush=True,
    )
    trainer.fit(lit, train_loader, val_loader)

    best = args.output_dir / "best.ckpt"
    target = args.output_dir / "best_checkpoint.pt"
    if best.exists():
        if target.exists():
            target.unlink()
        best.rename(target)


if __name__ == "__main__":
    main()

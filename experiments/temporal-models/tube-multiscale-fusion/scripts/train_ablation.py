"""Train the local-only 3D ResNet ablation classifier.

Reads a single named section from ``params.yaml`` (e.g. ``train_ablation_resnet3d``)
plus a ``local_branch`` section for the tube geometry.
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

from tube_multiscale_fusion.augment import build_tube_augment
from tube_multiscale_fusion.dataset import TubePatchDataset
from tube_multiscale_fusion.lit_ablation import LitAblationLocalResnet3D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True, help="Key in params.yaml")
    args = parser.parse_args()

    full_params = yaml.safe_load(args.params_path.read_text())
    cfg = full_params[args.params_key]
    augment_cfg = full_params.get("augment", {"enabled": False})
    l_cfg = full_params[cfg.get("local_branch_section", "local_branch")]

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(
        f"CUDA available: {torch.cuda.is_available()} | device: {device_name}",
        file=sys.stderr,
        flush=True,
    )

    L.seed_everything(cfg["seed"], workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = build_tube_augment(augment_cfg, train=True)
    val_transform = build_tube_augment(augment_cfg, train=False)

    train_ds = TubePatchDataset(
        args.train_dir, max_frames=cfg["max_frames"], transform=train_transform
    )
    val_ds = TubePatchDataset(
        args.val_dir, max_frames=cfg["max_frames"], transform=val_transform
    )

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

    lit = LitAblationLocalResnet3D(
        grid_size=l_cfg["grid_size"],
        cell_size=l_cfg["cell_size"],
        tube_length=l_cfg["tube_length"],
        temporal_stride=l_cfg["temporal_stride"],
        resnet_model=cfg.get("resnet_model", "r3d_18"),
        resnet_pretrained=cfg.get("resnet_pretrained", True),
        resnet_finetune=cfg.get("resnet_finetune", True),
        resnet_finetune_last_n_blocks=cfg.get("resnet_finetune_last_n_blocks", 1),
        resnet_clip_spatial_size=cfg.get("resnet_clip_spatial_size", 112),
        embed_dim=cfg.get("embed_dim"),
        head_hidden_dim=cfg["head_hidden_dim"],
        head_dropout=cfg.get("head_dropout", 0.1),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        backbone_lr=cfg.get("backbone_lr"),
        use_cosine_warmup=cfg.get("use_cosine_warmup", False),
        warmup_frac=cfg.get("warmup_frac", 0.05),
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
    trainer.fit(lit, train_loader, val_loader)

    best = args.output_dir / "best.ckpt"
    target = args.output_dir / "best_checkpoint.pt"
    if best.exists():
        if target.exists():
            target.unlink()
        best.rename(target)


if __name__ == "__main__":
    main()

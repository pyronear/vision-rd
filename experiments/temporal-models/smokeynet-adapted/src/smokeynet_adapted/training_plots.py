"""Per-run training curve plotting from Lightning CSVLogger metrics."""

import pandas as pd


def aggregate_train_loss_per_epoch(df: pd.DataFrame) -> pd.Series:
    train_rows = df.dropna(subset=["train/loss"])
    return train_rows.groupby("epoch")["train/loss"].mean()


_VAL_COLUMNS = [
    "epoch",
    "val/loss",
    "val/accuracy",
    "val/f1",
    "val/precision",
    "val/recall",
]


def extract_val_metrics_per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    val_rows = df.dropna(subset=["val/f1"])
    return val_rows[_VAL_COLUMNS].reset_index(drop=True)

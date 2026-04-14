"""Per-run training curve plotting from Lightning CSVLogger metrics."""

from pathlib import Path

import pandas as pd


def aggregate_train_loss_per_epoch(df: pd.DataFrame) -> pd.Series:
    train_rows = df.dropna(subset=["train/loss"])
    return train_rows.groupby("epoch")["train/loss"].mean()

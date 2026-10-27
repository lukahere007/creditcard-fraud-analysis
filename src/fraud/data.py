"""Loading the dataset and building model features."""

from pathlib import Path

import numpy as np
import pandas as pd

FULL_FILE = "creditcard.csv"
SAMPLE_FILE = "creditcard_sample.csv"
TARGET = "Class"


def resolve_data_path(data_dir: str | Path) -> Path:
    """Prefer the full Kaggle file when present, otherwise use the committed sample."""
    data_dir = Path(data_dir)
    full = data_dir / FULL_FILE
    return full if full.exists() else data_dir / SAMPLE_FILE


def load_data(data_dir: str | Path) -> pd.DataFrame:
    path = resolve_data_path(data_dir)
    df = pd.read_csv(path)
    df.attrs["source"] = path.name
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering. Stateless, so it is safe to apply before splitting.

    `Time` is seconds elapsed since the first transaction in a two-day window, so the
    hour derived from it is an *offset* from the start of recording, not wall-clock time.
    It is still cyclical, which is why it is encoded with sin/cos.
    """
    out = df.copy()
    hour = (out["Time"] // 3600) % 24
    out["HourSin"] = np.sin(2 * np.pi * hour / 24)
    out["HourCos"] = np.cos(2 * np.pi * hour / 24)
    out["LogAmount"] = np.log1p(out["Amount"])
    return out.drop(columns=["Time"])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target. `Amount` is kept aside for cost analysis."""
    X = df.drop(columns=[TARGET, "Amount"])
    y = df[TARGET].astype(int)
    return X, y

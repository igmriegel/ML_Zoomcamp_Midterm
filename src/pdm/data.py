"""Dataset loading, schema validation, and stratified splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from sklearn.model_selection import train_test_split

from pdm.config import (
    CSV_COLUMN_MAP,
    DATA_PATH,
    MODE_COLS,
    RAW_FEATURE_COLS,
    SEED,
    TARGET_COL,
)


class RawDatasetSchema(pa.DataFrameModel):
    """Pandera schema for the normalized AI4I 2020 dataset."""

    udi: Series[int] = pa.Field(ge=1, unique=True, nullable=False)
    product_id: Series[str] = pa.Field(str_matches=r"^[LMH]\d+$", nullable=False)
    type: Series[str] = pa.Field(isin=["L", "M", "H"], nullable=False)
    air_temperature_k: Series[float] = pa.Field(ge=290.0, le=310.0, nullable=False)
    process_temperature_k: Series[float] = pa.Field(ge=300.0, le=320.0, nullable=False)
    rotational_speed_rpm: Series[int] = pa.Field(ge=1000, le=3000, nullable=False)
    torque_nm: Series[float] = pa.Field(ge=0.0, le=100.0, nullable=False)
    tool_wear_min: Series[int] = pa.Field(ge=0, le=300, nullable=False)
    machine_failure: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    twf: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    hdf: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    pwf: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    osf: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    rnf: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        strict = True
        coerce = True


@dataclass(frozen=True)
class DatasetSplits:
    """Train, validation, and test splits for both binary and multi-label targets."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    y_modes_train: pd.DataFrame
    y_modes_val: pd.DataFrame
    y_modes_test: pd.DataFrame


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw CSV column names into API snake_case names."""
    return df.rename(columns=CSV_COLUMN_MAP)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate normalized dataset against the Pandera schema."""
    return RawDatasetSchema.validate(df, lazy=True)


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the AI4I 2020 dataset from CSV."""
    raw = pd.read_csv(path)
    normalized = rename_columns(raw)
    return validate_dataset(normalized)


def split_dataset(df: pd.DataFrame, seed: int = SEED) -> DatasetSplits:
    """Create stratified 60/20/20 train/val/test splits."""
    X = df[RAW_FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int)
    y_modes = df[MODE_COLS].astype(int)

    X_train_val, X_test, y_train_val, y_test, ym_train_val, ym_test = train_test_split(
        X,
        y,
        y_modes,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    X_train, X_val, y_train, y_val, ym_train, ym_val = train_test_split(
        X_train_val,
        y_train_val,
        ym_train_val,
        test_size=0.25,
        random_state=seed,
        stratify=y_train_val,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        y_modes_train=ym_train.reset_index(drop=True),
        y_modes_val=ym_val.reset_index(drop=True),
        y_modes_test=ym_test.reset_index(drop=True),
    )

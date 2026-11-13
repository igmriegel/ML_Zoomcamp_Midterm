"""Unit tests for dataset loading, validation, and splitting."""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from pdm.config import DATA_PATH, LEAKAGE_COLS, RAW_FEATURE_COLS
from pdm.data import load_raw, rename_columns, split_dataset, validate_dataset


def test_load_raw_validates_committed_csv() -> None:
    df = load_raw(DATA_PATH)
    assert len(df) == 10_000
    assert df.isna().sum().sum() == 0
    assert set(RAW_FEATURE_COLS).issubset(df.columns)


def test_rename_columns_supports_unit_suffix_variant() -> None:
    df = pd.DataFrame(
        {
            "UDI": [1],
            "Product ID": ["M14860"],
            "Type": ["M"],
            "Air temperature [K]": [298.1],
            "Process temperature [K]": [308.6],
            "Rotational speed [rpm]": [1551],
            "Torque [Nm]": [42.8],
            "Tool wear [min]": [0],
            "Machine failure": [0],
            "TWF": [0],
            "HDF": [0],
            "PWF": [0],
            "OSF": [0],
            "RNF": [0],
        }
    )
    renamed = validate_dataset(rename_columns(df))
    assert "air_temperature_k" in renamed.columns


def test_validate_dataset_rejects_invalid_type() -> None:
    df = load_raw(DATA_PATH).head(5).copy()
    df.loc[0, "type"] = "X"
    with pytest.raises(pa_errors.SchemaErrors):
        validate_dataset(df)


def test_validate_dataset_rejects_out_of_range_torque() -> None:
    df = load_raw(DATA_PATH).head(5).copy()
    df.loc[0, "torque_nm"] = -5.0
    with pytest.raises(pa_errors.SchemaErrors):
        validate_dataset(df)


def test_split_dataset_preserves_stratification_and_excludes_leakage() -> None:
    df = load_raw(DATA_PATH)
    splits = split_dataset(df)

    assert len(splits.X_train) == 6_000
    assert len(splits.X_val) == 2_000
    assert len(splits.X_test) == 2_000

    base_rate = df["machine_failure"].mean()
    assert abs(splits.y_train.mean() - base_rate) < 0.005
    assert abs(splits.y_val.mean() - base_rate) < 0.005
    assert abs(splits.y_test.mean() - base_rate) < 0.005

    for col in LEAKAGE_COLS:
        assert col not in splits.X_train.columns

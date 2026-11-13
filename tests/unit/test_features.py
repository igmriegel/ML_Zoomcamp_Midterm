"""Unit tests for physical feature engineering."""

from __future__ import annotations

import math

import pandas as pd

from pdm.config import LEAKAGE_COLS, RAW_FEATURE_COLS
from pdm.features import add_features
from pdm.pipeline import build_preprocessor


def test_add_features_computes_physical_quantities_and_does_not_mutate() -> None:
    df = pd.DataFrame(
        {
            "type": ["M"],
            "air_temperature_k": [298.0],
            "process_temperature_k": [308.0],
            "rotational_speed_rpm": [1500.0],
            "torque_nm": [40.0],
            "tool_wear_min": [10.0],
        }
    )
    original = df.copy()
    enriched = add_features(df)

    pd.testing.assert_frame_equal(df, original)
    assert math.isclose(enriched["temp_diff_k"].iloc[0], 10.0)
    assert math.isclose(enriched["power_w"].iloc[0], 40.0 * 1500.0 * 2.0 * math.pi / 60.0)
    assert math.isclose(enriched["strain"].iloc[0], 400.0)
    assert math.isclose(enriched["strain_ratio"].iloc[0], 400.0 / 12000.0)


def test_preprocessor_drops_leakage_columns() -> None:
    df = pd.DataFrame(
        {
            "type": ["L", "M"],
            "air_temperature_k": [298.0, 299.0],
            "process_temperature_k": [308.0, 309.0],
            "rotational_speed_rpm": [1500.0, 1600.0],
            "torque_nm": [40.0, 45.0],
            "tool_wear_min": [10.0, 20.0],
            "udi": [1, 2],
            "product_id": ["L1", "M2"],
            "twf": [0, 1],
            "hdf": [0, 0],
            "pwf": [0, 0],
            "osf": [0, 0],
            "rnf": [0, 0],
        }
    )
    enriched = add_features(df)
    prep = build_preprocessor()
    prep.fit(enriched)
    feature_names = list(prep.get_feature_names_out())

    for col in LEAKAGE_COLS:
        assert all(col not in name for name in feature_names)
    assert len(RAW_FEATURE_COLS) == 6

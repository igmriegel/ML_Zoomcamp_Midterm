"""Pure feature engineering transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed physical features without mutating the input DataFrame.

    - temp_diff_k: temperature gradient linked to heat dissipation failures (HDF).
    - power_w: mechanical power in Watts = torque * angular velocity (PWF).
    - strain: cumulative tool wear * torque proxy for overstrain (OSF).
    """
    out = df.copy()
    out["temp_diff_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * (2.0 * np.pi / 60.0)
    out["strain"] = out["tool_wear_min"] * out["torque_nm"]
    return out

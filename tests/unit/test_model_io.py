"""Unit tests for ModelBundle serialization and Pydantic schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pdm.config import DATA_PATH, RAW_FEATURE_COLS, TARGET_COL
from pdm.data import load_raw
from pdm.model_io import ModelBundle, load_bundle, save_bundle
from pdm.pipeline import build_mode_pipeline, build_pipeline, get_model_candidates
from pdm.schemas import BatchRequest, MachineReading


def test_model_bundle_roundtrip(tmp_path: Path) -> None:
    df = load_raw(DATA_PATH).sample(n=800, random_state=42)
    est, _ = get_model_candidates(seed=42)["logreg"]
    bin_pipe = build_pipeline(est).fit(df[RAW_FEATURE_COLS], df[TARGET_COL])
    mode_pipe = build_mode_pipeline(est).fit(df[RAW_FEATURE_COLS], df[["twf", "hdf", "pwf", "osf"]])

    bundle: ModelBundle = {
        "version": "0.1.0",
        "trained_at": "2026-11-13T00:00:00Z",
        "algorithm": "logreg",
        "binary": bin_pipe,
        "threshold": 0.25,
        "modes": mode_pipe,
        "mode_thresholds": {"twf": 0.5, "hdf": 0.5, "pwf": 0.5, "osf": 0.5},
        "features": RAW_FEATURE_COLS,
    }
    out = tmp_path / "model.bin"
    save_bundle(bundle, out)
    loaded = load_bundle(out)

    assert loaded["version"] == "0.1.0"
    assert loaded["algorithm"] == "logreg"
    assert loaded["threshold"] == 0.25
    assert loaded["binary"].predict_proba(df[RAW_FEATURE_COLS].head(2)).shape == (2, 2)


def test_machine_reading_schema_validation() -> None:
    valid = MachineReading(
        type="M",
        air_temperature_k=298.1,
        process_temperature_k=308.6,
        rotational_speed_rpm=1551.0,
        torque_nm=42.8,
        tool_wear_min=0.0,
    )
    assert valid.type == "M"

    with pytest.raises(ValidationError):
        MachineReading(
            type="X",  # type: ignore[arg-type]
            air_temperature_k=298.1,
            process_temperature_k=308.6,
            rotational_speed_rpm=1551.0,
            torque_nm=42.8,
            tool_wear_min=0.0,
        )

    with pytest.raises(ValidationError):
        BatchRequest(readings=[])

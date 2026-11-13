"""Pydantic request and response schemas for the FastAPI service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MachineReading(BaseModel):
    """Single sensor reading payload matching physical ranges."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(ge=290.0, le=310.0)
    process_temperature_k: float = Field(ge=300.0, le=320.0)
    rotational_speed_rpm: float = Field(ge=1000.0, le=3000.0)
    torque_nm: float = Field(ge=0.0, le=100.0)
    tool_wear_min: float = Field(ge=0.0, le=300.0)


class BatchRequest(BaseModel):
    """Batch prediction payload (1 to 1000 readings)."""

    model_config = ConfigDict(extra="forbid")

    readings: list[MachineReading] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Prediction response for a single machine reading."""

    failure_probability: float
    failure: bool
    threshold: float
    mode_probabilities: dict[str, float]
    likely_modes: list[str]
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_version: str
    algorithm: str

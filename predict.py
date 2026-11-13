"""FastAPI inference service for machine failure and failure-mode prediction."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI

from pdm.config import MODE_COLS, MODEL_PATH
from pdm.model_io import ModelBundle, load_bundle
from pdm.schemas import (
    BatchPredictionResponse,
    BatchRequest,
    HealthResponse,
    MachineReading,
    PredictionResponse,
)


def _load_configured_bundle() -> ModelBundle:
    path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    return load_bundle(path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the serialized ModelBundle once on startup."""
    app.state.bundle = _load_configured_bundle()
    yield


app = FastAPI(
    title="Predictive Maintenance API (AI4I 2020)",
    description=(
        "Predicts machine failure probability and likely failure modes (TWF, HDF, PWF, OSF)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _get_bundle() -> ModelBundle:
    if not hasattr(app.state, "bundle"):
        app.state.bundle = _load_configured_bundle()
    bundle: ModelBundle = app.state.bundle
    return bundle


def _predict_df(bundle: ModelBundle, df: pd.DataFrame) -> list[PredictionResponse]:
    threshold = float(bundle["threshold"])
    bin_probs = bundle["binary"].predict_proba(df)[:, 1]

    raw_mode_probs = bundle["modes"].predict_proba(df)
    mode_matrix = np.column_stack(
        [p[:, 1] if p.shape[1] == 2 else np.zeros(len(p), dtype=float) for p in raw_mode_probs]
    )

    responses: list[PredictionResponse] = []
    for i in range(len(df)):
        fail_prob = float(bin_probs[i])
        is_failure = fail_prob >= threshold
        mode_probs = {mode: float(mode_matrix[i, idx]) for idx, mode in enumerate(MODE_COLS)}
        likely_modes: list[str] = []
        if is_failure:
            likely_modes = sorted(
                [m for m, p in mode_probs.items() if p >= bundle["mode_thresholds"].get(m, 0.5)],
                key=lambda m: mode_probs[m],
                reverse=True,
            )

        responses.append(
            PredictionResponse(
                failure_probability=round(fail_prob, 6),
                failure=is_failure,
                threshold=threshold,
                mode_probabilities={k: round(v, 6) for k, v in mode_probs.items()},
                likely_modes=likely_modes,
                model_version=bundle["version"],
            )
        )
    return responses


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health and loaded model metadata."""
    bundle = _get_bundle()
    return HealthResponse(
        status="ok",
        model_version=bundle["version"],
        algorithm=bundle["algorithm"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: MachineReading) -> PredictionResponse:
    """Predict machine failure and failure modes for a single sensor reading."""
    bundle = _get_bundle()
    df = pd.DataFrame([reading.model_dump()])
    return _predict_df(bundle, df)[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchRequest) -> BatchPredictionResponse:
    """Predict machine failure and failure modes for a batch of readings."""
    bundle = _get_bundle()
    df = pd.DataFrame([item.model_dump() for item in payload.readings])
    return BatchPredictionResponse(predictions=_predict_df(bundle, df))

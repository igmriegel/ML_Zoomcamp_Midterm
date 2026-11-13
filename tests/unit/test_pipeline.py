"""Unit tests for binary and multi-output ML pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.config import DATA_PATH, MODE_COLS, RAW_FEATURE_COLS, TARGET_COL
from pdm.data import load_raw
from pdm.pipeline import build_mode_pipeline, build_pipeline, get_model_candidates


def test_all_model_candidates_fit_and_predict_proba() -> None:
    df = load_raw(DATA_PATH).head(300)
    X = df[RAW_FEATURE_COLS]
    y = df[TARGET_COL]

    candidates = get_model_candidates(seed=42, scale_pos_weight=10.0)
    assert set(candidates.keys()) == {"logreg", "random_forest", "xgboost", "lightgbm"}

    for name, (estimator, search_space) in candidates.items():
        assert len(search_space) >= 2
        pipe = build_pipeline(estimator)
        pipe.fit(X, y)
        probs = pipe.predict_proba(X)[:, 1]
        assert probs.shape == (300,), name
        assert np.all((probs >= 0.0) & (probs <= 1.0)), name


def test_mode_pipeline_outputs_four_probabilities() -> None:
    df = load_raw(DATA_PATH).head(300)
    X = df[RAW_FEATURE_COLS]
    y_modes = df[MODE_COLS]

    est, _ = get_model_candidates(seed=42)["random_forest"]
    mode_pipe = build_mode_pipeline(est)
    mode_pipe.fit(X, y_modes)

    raw_probs = mode_pipe.predict_proba(X.head(5))
    assert len(raw_probs) == 4
    stacked = np.column_stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(p)) for p in raw_probs])
    assert stacked.shape == (5, 4)
    assert np.all((stacked >= 0.0) & (stacked <= 1.0))


def test_pipeline_handles_unseen_type_category() -> None:
    df = load_raw(DATA_PATH).head(100)
    X = df[RAW_FEATURE_COLS]
    y = df[TARGET_COL]

    est, _ = get_model_candidates(seed=42)["logreg"]
    pipe = build_pipeline(est)
    pipe.fit(X, y)

    unseen = pd.DataFrame(
        {
            "type": ["Z"],
            "air_temperature_k": [298.0],
            "process_temperature_k": [308.0],
            "rotational_speed_rpm": [1500.0],
            "torque_nm": [40.0],
            "tool_wear_min": [10.0],
        }
    )
    prob = pipe.predict_proba(unseen)[:, 1]
    assert 0.0 <= prob[0] <= 1.0

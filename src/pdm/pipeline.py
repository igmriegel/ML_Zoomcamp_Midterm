"""Model pipelines, hyperparameter search spaces, and multi-label wrapper."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from pdm.config import CATEGORICAL_COLS, NUMERIC_COLS, SEED
from pdm.features import add_features

ModelFamily = str


def build_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Build end-to-end sklearn Pipeline: feature engineering -> preprocess -> estimator."""
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def build_mode_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Build multi-output pipeline for failure modes (TWF, HDF, PWF, OSF)."""
    return build_pipeline(MultiOutputClassifier(clone(estimator)))


def get_model_candidates(
    seed: int = SEED, scale_pos_weight: float = 1.0
) -> dict[ModelFamily, tuple[BaseEstimator, dict[str, list[Any]]]]:
    """Return candidate estimators and RandomizedSearchCV parameter distributions."""
    return {
        "logreg": (
            LogisticRegression(max_iter=1000, random_state=seed),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0, 50.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [6, 10, 14, None],
                "model__min_samples_leaf": [1, 2, 5],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=seed,
                n_jobs=1,
                eval_metric="logloss",
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__scale_pos_weight": [1.0, scale_pos_weight],
            },
        ),
        "lightgbm": (
            LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1),
            {
                "model__n_estimators": [100, 200, 300],
                "model__num_leaves": [15, 31, 63],
                "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "model__min_child_samples": [10, 20, 40],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }

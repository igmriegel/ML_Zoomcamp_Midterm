"""Serialization and loading for the versioned ModelBundle artifact."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import joblib
from sklearn.pipeline import Pipeline


class ModelBundle(TypedDict):
    """Self-contained artifact holding binary and multi-label pipelines and metadata."""

    version: str
    trained_at: str
    algorithm: str
    binary: Pipeline
    threshold: float
    modes: Pipeline
    mode_thresholds: dict[str, float]
    features: list[str]


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    """Persist ModelBundle to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ModelBundle:
    """Load ModelBundle from disk."""
    bundle: ModelBundle = joblib.load(path)
    return bundle

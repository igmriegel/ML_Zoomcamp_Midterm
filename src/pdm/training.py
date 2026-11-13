"""Shared training, hyperparameter tuning, and evaluation workflow."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from pdm.config import (
    COST_FN,
    COST_FP,
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    RAW_FEATURE_COLS,
    SEED,
)
from pdm.data import DatasetSplits, load_raw, split_dataset
from pdm.evaluate import binary_metrics, choose_mode_thresholds, choose_threshold, mode_metrics
from pdm.model_io import ModelBundle, save_bundle
from pdm.pipeline import build_mode_pipeline, build_pipeline, get_model_candidates


@dataclass
class CandidateResult:
    """Tuning and validation summary for one model family."""

    family: str
    cv_pr_auc: float
    val_metrics: dict[str, float]
    best_params: dict[str, Any]
    best_pipeline: Any
    fit_seconds: float


def _extract_mode_probs(mode_pipe: Any, X: pd.DataFrame) -> np.ndarray:
    """Extract (n_samples, n_modes) probability matrix from MultiOutputClassifier."""
    raw_probs = mode_pipe.predict_proba(X)
    return np.column_stack(
        [p[:, 1] if p.shape[1] == 2 else np.zeros(len(p), dtype=float) for p in raw_probs]
    )


def _make_mode_estimator(winning_estimator: BaseEstimator) -> BaseEstimator:
    """Clone winning estimator with class balancing for rare failure modes."""
    est = clone(winning_estimator)
    if isinstance(est, RandomForestClassifier):
        est.set_params(class_weight="balanced_subsample")
    elif "class_weight" in est.get_params():
        est.set_params(class_weight="balanced")
    return est


def tune_candidates(
    splits: DatasetSplits,
    fast: bool = False,
    seed: int = SEED,
) -> tuple[dict[str, CandidateResult], str]:
    """Tune all candidate families on X_train via StratifiedKFold and evaluate on X_val."""
    neg_count = int((splits.y_train == 0).sum())
    pos_count = max(int((splits.y_train == 1).sum()), 1)
    scale_pos_weight = float(neg_count / pos_count)

    candidates = get_model_candidates(seed=seed, scale_pos_weight=scale_pos_weight)
    n_splits = 3 if fast else 5
    n_iter = 3 if fast else 15
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: dict[str, CandidateResult] = {}
    for family, (estimator, param_dist) in candidates.items():
        t0 = time.perf_counter()
        pipe = build_pipeline(estimator)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=min(n_iter, _max_combinations(param_dist)),
            scoring="average_precision",
            cv=cv,
            random_state=seed,
            n_jobs=1,
            refit=True,
        )
        search.fit(splits.X_train, splits.y_train)
        elapsed = time.perf_counter() - t0

        val_probs = search.best_estimator_.predict_proba(splits.X_val)[:, 1]
        val_thr, _ = choose_threshold(splits.y_val, val_probs, cost_fn=COST_FN, cost_fp=COST_FP)
        val_met = binary_metrics(splits.y_val, val_probs, threshold=val_thr)

        results[family] = CandidateResult(
            family=family,
            cv_pr_auc=float(search.best_score_),
            val_metrics=val_met,
            best_params=dict(search.best_params_),
            best_pipeline=search.best_estimator_,
            fit_seconds=round(elapsed, 3),
        )

    winner = max(results, key=lambda k: results[k].cv_pr_auc)
    return results, winner


def _max_combinations(param_dist: dict[str, list[Any]]) -> int:
    total = 1
    for vals in param_dist.values():
        total *= len(vals)
    return total


def train_and_evaluate(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    fast: bool = False,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run full training pipeline, save ModelBundle and test metrics."""
    df = load_raw(data_path)
    splits = split_dataset(df, seed=seed)

    results, winner_family = tune_candidates(splits, fast=fast, seed=seed)
    winner_res = results[winner_family]

    # Select cost-optimal threshold strictly on validation set
    val_probs = winner_res.best_pipeline.predict_proba(splits.X_val)[:, 1]
    threshold, _ = choose_threshold(splits.y_val, val_probs, cost_fn=COST_FN, cost_fp=COST_FP)

    # Refit winning binary pipeline and logreg baseline on train + val
    X_train_val = pd.concat([splits.X_train, splits.X_val], ignore_index=True)
    y_train_val = pd.concat([splits.y_train, splits.y_val], ignore_index=True)
    y_modes_train_val = pd.concat([splits.y_modes_train, splits.y_modes_val], ignore_index=True)

    final_binary = clone(winner_res.best_pipeline)
    final_binary.fit(X_train_val, y_train_val)

    baseline_pipe = clone(results["logreg"].best_pipeline)
    baseline_pipe.fit(X_train_val, y_train_val)

    # Train multi-label failure mode model with validation-calibrated thresholds
    mode_estimator = _make_mode_estimator(final_binary.named_steps["model"])
    val_modes = build_mode_pipeline(mode_estimator)
    val_modes.fit(splits.X_train, splits.y_modes_train)
    val_mode_probs = _extract_mode_probs(val_modes, splits.X_val)
    mode_thresholds = choose_mode_thresholds(splits.y_modes_val, val_mode_probs, beta=1.0)

    final_modes = build_mode_pipeline(mode_estimator)
    final_modes.fit(X_train_val, y_modes_train_val)

    # Single final evaluation on held-out test set
    test_probs = final_binary.predict_proba(splits.X_test)[:, 1]
    test_binary = binary_metrics(splits.y_test, test_probs, threshold=threshold)

    baseline_test_probs = baseline_pipe.predict_proba(splits.X_test)[:, 1]
    baseline_test = binary_metrics(splits.y_test, baseline_test_probs, threshold=0.5)

    test_mode_probs = _extract_mode_probs(final_modes, splits.X_test)
    test_modes = mode_metrics(splits.y_modes_test, test_mode_probs, thresholds=mode_thresholds)

    bundle: ModelBundle = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "algorithm": winner_family,
        "binary": final_binary,
        "threshold": threshold,
        "modes": final_modes,
        "mode_thresholds": mode_thresholds,
        "features": RAW_FEATURE_COLS,
    }
    save_bundle(bundle, model_path)

    metrics_payload: dict[str, Any] = {
        "version": MODEL_VERSION,
        "seed": seed,
        "fast": fast,
        "winner": winner_family,
        "threshold": threshold,
        "mode_thresholds": mode_thresholds,
        "candidates_cv": {
            fam: {
                "cv_pr_auc": res.cv_pr_auc,
                "val_pr_auc": res.val_metrics["pr_auc"],
                "val_roc_auc": res.val_metrics["roc_auc"],
                "fit_seconds": res.fit_seconds,
                "best_params": res.best_params,
            }
            for fam, res in results.items()
        },
        "test_binary": test_binary,
        "baseline_logreg_test_pr_auc": baseline_test["pr_auc"],
        "test_modes": test_modes,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload

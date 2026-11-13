"""Evaluation metrics and cost-based threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pdm.config import COST_FN, COST_FP, MODE_COLS


def expected_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Compute total misclassification cost (FN * cost_fn + FP * cost_fp)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    _ = tn, tp
    return float(fn * cost_fn + fp * cost_fp)


def choose_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[float, float]:
    """Select probability threshold in [0.01, 0.99] minimizing expected cost."""
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(y_prob)
    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.50
    best_cost = float("inf")
    for thr in thresholds:
        preds = (p_arr >= thr).astype(int)
        cost = expected_cost(y_arr, preds, cost_fn=cost_fn, cost_fp=cost_fp)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(round(float(thr), 2))
    return best_threshold, best_cost


def binary_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    threshold: float,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> dict[str, float]:
    """Compute binary classification metrics at a given threshold."""
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(y_prob)
    y_pred = (p_arr >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_arr, p_arr)),
        "roc_auc": float(roc_auc_score(y_arr, p_arr)),
        "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
        "cost": expected_cost(y_arr, y_pred, cost_fn=cost_fn, cost_fp=cost_fp),
    }


def choose_mode_thresholds(
    y_true_modes: pd.DataFrame,
    y_prob_modes: np.ndarray,
    mode_cols: list[str] = MODE_COLS,
    beta: float = 1.0,
) -> dict[str, float]:
    """Select per-mode probability thresholds maximizing F-beta score on validation."""
    thresholds = np.linspace(0.05, 0.95, 91)
    selected: dict[str, float] = {}

    for idx, mode in enumerate(mode_cols):
        y_m = y_true_modes[mode].to_numpy()
        p_m = y_prob_modes[:, idx]
        if int(y_m.sum()) == 0:
            selected[mode] = 0.5
            continue

        best_thr = 0.5
        best_score = -1.0
        for thr in thresholds:
            preds = (p_m >= thr).astype(int)
            score = float(fbeta_score(y_m, preds, beta=beta, zero_division=0))
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)
            ):
                best_score = score
                best_thr = float(round(float(thr), 2))
        selected[mode] = best_thr

    return selected


def mode_metrics(
    y_true_modes: pd.DataFrame,
    y_prob_modes: np.ndarray,
    thresholds: dict[str, float] | None = None,
    mode_cols: list[str] = MODE_COLS,
) -> dict[str, object]:
    """Compute per-mode and macro PR-AUC, F1, precision, and recall."""
    per_mode: dict[str, dict[str, float]] = {}
    pr_aucs: list[float] = []
    f1s: list[float] = []
    recalls: list[float] = []
    prevalences: list[float] = []

    for idx, mode in enumerate(mode_cols):
        y_m = y_true_modes[mode].to_numpy()
        p_m = y_prob_modes[:, idx]
        thr = 0.5 if thresholds is None else thresholds.get(mode, 0.5)
        y_pred = (p_m >= thr).astype(int)

        pr_auc = float(average_precision_score(y_m, p_m))
        f1 = float(f1_score(y_m, y_pred, zero_division=0))
        prec = float(precision_score(y_m, y_pred, zero_division=0))
        rec = float(recall_score(y_m, y_pred, zero_division=0))
        prev = float(y_m.mean())

        per_mode[mode] = {
            "pr_auc": pr_auc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "prevalence": prev,
            "threshold": float(thr),
        }
        pr_aucs.append(pr_auc)
        f1s.append(f1)
        recalls.append(rec)
        prevalences.append(prev)

    return {
        "per_mode": per_mode,
        "macro_pr_auc": float(np.mean(pr_aucs)),
        "macro_f1": float(np.mean(f1s)),
        "macro_recall": float(np.mean(recalls)),
        "mean_prevalence": float(np.mean(prevalences)),
    }

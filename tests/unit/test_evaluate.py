"""Unit tests for evaluation metrics and cost-based threshold optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.evaluate import binary_metrics, choose_threshold, expected_cost, mode_metrics


def test_expected_cost_weights_fn_and_fp() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])  # 1 FN, 1 FP
    assert expected_cost(y_true, y_pred, cost_fn=10.0, cost_fp=1.0) == 11.0


def test_choose_threshold_favors_lower_threshold_when_fn_is_costly() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.25, 0.30, 0.35, 0.80])

    threshold, cost = choose_threshold(y_true, y_prob, cost_fn=10.0, cost_fp=1.0)
    assert 0.31 <= threshold <= 0.35
    assert cost == 0.0


def test_binary_metrics_returns_expected_keys() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    assert set(metrics.keys()) == {"pr_auc", "roc_auc", "f1", "precision", "recall", "cost"}
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["cost"] == 0.0


def test_mode_metrics_computes_per_mode_and_macro() -> None:
    y_modes = pd.DataFrame(
        {
            "twf": [0, 1, 0, 0],
            "hdf": [1, 0, 0, 0],
            "pwf": [0, 0, 1, 0],
            "osf": [0, 0, 0, 1],
        }
    )
    y_prob = np.array(
        [
            [0.1, 0.9, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.1],
            [0.1, 0.1, 0.1, 0.9],
        ]
    )
    res = mode_metrics(y_modes, y_prob)
    assert res["macro_pr_auc"] == 1.0
    assert res["macro_recall"] == 1.0

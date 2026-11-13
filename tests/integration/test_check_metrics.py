"""Integration tests for scripts/check_metrics.py quality gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_metrics import check_metrics


def _good_metrics() -> dict[str, object]:
    return {
        "test_binary": {"pr_auc": 0.88, "roc_auc": 0.98},
        "baseline_logreg_test_pr_auc": 0.50,
        "test_modes": {
            "macro_pr_auc": 0.70,
            "mean_prevalence": 0.01,
            "per_mode": {
                "twf": {"recall": 0.10},
                "hdf": {"recall": 0.80},
                "pwf": {"recall": 0.85},
                "osf": {"recall": 0.80},
            },
        },
    }


def test_check_metrics_passes_on_valid_payload() -> None:
    assert check_metrics(_good_metrics()) == []


def test_check_metrics_fails_on_poor_binary_and_mode_metrics(tmp_path: Path) -> None:
    bad = _good_metrics()
    bad["test_binary"] = {"pr_auc": 0.40, "roc_auc": 0.80}
    bad["test_modes"] = {
        "macro_pr_auc": 0.005,
        "mean_prevalence": 0.01,
        "per_mode": {
            "twf": {"recall": 0.0},
            "hdf": {"recall": 0.2},
            "pwf": {"recall": 0.2},
            "osf": {"recall": 0.2},
        },
    }
    errors = check_metrics(bad)
    assert len(errors) >= 5

    bad_file = tmp_path / "bad_metrics.json"
    bad_file.write_text(json.dumps(bad), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/check_metrics.py", "--metrics", str(bad_file)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "FAIL:" in proc.stderr

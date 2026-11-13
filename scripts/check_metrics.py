"""Quality gate verifying test-set metrics meet minimum thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pdm.config import METRICS_PATH

MIN_PR_AUC = 0.75
MIN_ROC_AUC = 0.95
MIN_CRITICAL_MODE_RECALL = 0.55
CRITICAL_MODES = ("hdf", "pwf", "osf")


def check_metrics(metrics: dict[str, Any]) -> list[str]:
    """Return list of quality gate failure reasons (empty if all pass)."""
    errors: list[str] = []
    tb = metrics["test_binary"]

    if tb["pr_auc"] < MIN_PR_AUC:
        errors.append(f"Binary PR-AUC {tb['pr_auc']:.4f} < {MIN_PR_AUC}")
    if tb["roc_auc"] < MIN_ROC_AUC:
        errors.append(f"Binary ROC-AUC {tb['roc_auc']:.4f} < {MIN_ROC_AUC}")
    if tb["pr_auc"] <= metrics["baseline_logreg_test_pr_auc"]:
        errors.append(
            f"Winner PR-AUC ({tb['pr_auc']:.4f}) did not beat LogReg baseline "
            f"({metrics['baseline_logreg_test_pr_auc']:.4f})"
        )

    tm = metrics["test_modes"]
    if tm["macro_pr_auc"] <= tm["mean_prevalence"]:
        errors.append(
            f"Mode macro PR-AUC ({tm['macro_pr_auc']:.4f}) <= mean prevalence "
            f"({tm['mean_prevalence']:.4f})"
        )

    for mode in CRITICAL_MODES:
        rec = tm["per_mode"][mode]["recall"]
        if rec < MIN_CRITICAL_MODE_RECALL:
            errors.append(f"Mode '{mode}' recall {rec:.4f} < {MIN_CRITICAL_MODE_RECALL}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trained model quality gate.")
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH, help="Path to metrics.json.")
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    errors = check_metrics(payload)
    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        sys.exit(1)

    print("Quality gate PASSED.")


if __name__ == "__main__":
    main()

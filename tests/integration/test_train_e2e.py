"""End-to-end integration test for train.py CLI and quality gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_metrics import check_metrics


def test_train_cli_e2e_and_quality_gate(tmp_path: Path) -> None:
    model_path = tmp_path / "model.bin"
    metrics_path = tmp_path / "metrics.json"

    proc = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--fast",
            "--out",
            str(model_path),
            "--metrics",
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Winner:" in proc.stdout
    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    errors = check_metrics(metrics)
    assert errors == []

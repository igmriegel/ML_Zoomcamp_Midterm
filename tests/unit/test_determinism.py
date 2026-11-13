"""Determinism test: identical seed produces identical metrics."""

from __future__ import annotations

from pathlib import Path

from pdm.config import DATA_PATH
from pdm.training import train_and_evaluate


def test_training_is_deterministic_with_fixed_seed(tmp_path: Path) -> None:
    m1 = train_and_evaluate(
        data_path=DATA_PATH,
        model_path=tmp_path / "m1.bin",
        metrics_path=tmp_path / "met1.json",
        fast=True,
        seed=42,
    )
    m2 = train_and_evaluate(
        data_path=DATA_PATH,
        model_path=tmp_path / "m2.bin",
        metrics_path=tmp_path / "met2.json",
        fast=True,
        seed=42,
    )
    assert m1["winner"] == m2["winner"]
    assert m1["threshold"] == m2["threshold"]
    assert m1["test_binary"] == m2["test_binary"]
    assert m1["test_modes"] == m2["test_modes"]

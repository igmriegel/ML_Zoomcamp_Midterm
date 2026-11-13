"""API contract and behavioral tests for FastAPI service."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdm.config import DATA_PATH
from pdm.training import train_and_evaluate
from predict import app

NOMINAL = {
    "type": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551.0,
    "torque_nm": 42.8,
    "tool_wear_min": 0.0,
}

HIGH_RISK = {
    "type": "L",
    "air_temperature_k": 302.5,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1320.0,
    "torque_nm": 68.5,
    "tool_wear_min": 215.0,
}


@pytest.fixture(scope="module")
def api_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    model_dir = tmp_path_factory.mktemp("api_model")
    model_path = model_dir / "model.bin"
    metrics_path = model_dir / "metrics.json"
    train_and_evaluate(
        data_path=DATA_PATH,
        model_path=model_path,
        metrics_path=metrics_path,
        fast=True,
    )
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(model_path))
    if hasattr(app.state, "bundle"):
        delattr(app.state, "bundle")
    with TestClient(app) as client:
        yield client
    mp.undo()


def test_health_endpoint(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert body["algorithm"] in {"logreg", "random_forest", "xgboost", "lightgbm"}


def test_predict_nominal_and_high_risk_monotonicity(api_client: TestClient) -> None:
    r_nom = api_client.post("/predict", json=NOMINAL)
    r_risk = api_client.post("/predict", json=HIGH_RISK)
    assert r_nom.status_code == 200
    assert r_risk.status_code == 200

    nom = r_nom.json()
    risk = r_risk.json()

    assert 0.0 <= nom["failure_probability"] <= 1.0
    assert 0.0 <= risk["failure_probability"] <= 1.0
    assert risk["failure_probability"] > nom["failure_probability"]
    assert set(nom["mode_probabilities"].keys()) == {"twf", "hdf", "pwf", "osf"}

    if not nom["failure"]:
        assert nom["likely_modes"] == []


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},  # missing fields
        {**NOMINAL, "type": "X"},  # invalid literal
        {**NOMINAL, "torque_nm": -1.0},  # below range
        {**NOMINAL, "rotational_speed_rpm": 5000.0},  # above range
        {**NOMINAL, "extra_field": 123},  # extra="forbid"
    ],
)
def test_predict_rejects_invalid_payloads_with_422(
    api_client: TestClient, bad_payload: dict[str, object]
) -> None:
    resp = api_client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_batch_valid_and_invalid(api_client: TestClient) -> None:
    resp = api_client.post("/predict/batch", json={"readings": [NOMINAL, HIGH_RISK]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2

    empty = api_client.post("/predict/batch", json={"readings": []})
    assert empty.status_code == 422


def test_predict_lazy_bundle_load_without_lifespan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.bin"
    metrics_path = tmp_path / "metrics.json"
    train_and_evaluate(
        data_path=DATA_PATH,
        model_path=model_path,
        metrics_path=metrics_path,
        fast=True,
    )
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    if hasattr(app.state, "bundle"):
        delattr(app.state, "bundle")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200

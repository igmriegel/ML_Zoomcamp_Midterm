"""Example client script to test the deployed /predict and /health endpoints."""

from __future__ import annotations

import argparse
import json
import sys

import requests

NOMINAL_READING = {
    "type": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551.0,
    "torque_nm": 42.8,
    "tool_wear_min": 0.0,
}

HIGH_RISK_READING = {
    "type": "L",
    "air_temperature_k": 302.5,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1320.0,
    "torque_nm": 68.5,
    "tool_wear_min": 215.0,
}


def run_smoke(base_url: str, timeout: float = 60.0) -> None:
    """Call /health and /predict with nominal and high-risk payloads."""
    base = base_url.rstrip("/")

    health = requests.get(f"{base}/health", timeout=timeout)
    health.raise_for_status()
    print("GET /health ->", json.dumps(health.json(), indent=2))

    for label, payload in [("nominal", NOMINAL_READING), ("high_risk", HIGH_RISK_READING)]:
        resp = requests.post(f"{base}/predict", json=payload, timeout=timeout)
        resp.raise_for_status()
        print(f"POST /predict ({label}) ->", json.dumps(resp.json(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Predictive Maintenance API.")
    parser.add_argument(
        "--url",
        default="http://localhost:9696",
        help="Base URL of the running API (local or Render).",
    )
    args = parser.parse_args()

    try:
        run_smoke(args.url)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Predictive Maintenance — Machine Failure & Failure Mode Prediction (AI4I 2020)

[![CI](https://github.com/igmriegel/ML_Zoomcamp_Midterm/actions/workflows/ci.yml/badge.svg)](https://github.com/igmriegel/ML_Zoomcamp_Midterm/actions/workflows/ci.yml)

End-to-end machine learning service for industrial **predictive maintenance** on a milling machine, built for the ML Zoomcamp Midterm Project.

Given real-time sensor readings, the API answers two operational questions:
1. **Will the machine fail?** (`failure`: binary classification with a cost-optimal probability threshold where $\text{Cost}_{\text{FN}} = 10 \times \text{Cost}_{\text{FP}}$).
2. **Which failure mode(s) are likely?** (`likely_modes`: multi-label classification across `TWF`, `HDF`, `PWF`, and `OSF`).

---

## 1. Problem Description & Business Context

Unplanned milling machine downtime causes costly production halts, scrap parts, and potential equipment damage, whereas unnecessary preventive inspections incur a much smaller operational cost. We model this asymmetry explicitly using a **10:1 cost ratio** ($\text{FN} = 10$, $\text{FP} = 1$).

- **Target users:** Plant reliability engineers, maintenance planners, and automated SCADA/IoT alerting pipelines.
- **Primary selection metric:** **PR-AUC** (Average Precision) via 5-fold stratified cross-validation, because machine failures are rare (~3.39% positive rate) and PR-AUC focuses on precision/recall trade-offs on the minority failure class.
- **Operating threshold:** Selected on the validation split to minimize total expected misclassification cost, then evaluated **once** on the held-out test split.

---

## 2. Dataset & EDA Summary

- **Source:** [AI4I 2020 Predictive Maintenance Dataset (UCI ID 601)](https://archive.ics.uci.edu/dataset/601) — 10,000 rows, 14 columns, CC BY 4.0 license, committed in [`data/ai4i2020.csv`](data/ai4i2020.csv) (with [`scripts/download_data.py`](scripts/download_data.py) as fallback).
- **Data integrity:** 0 missing values, 0 duplicate rows, unique `udi` identifiers, validated with a strict `pandera` schema in [`src/pdm/data.py`](src/pdm/data.py).
- **Leakage prevention:** `udi`, `product_id`, and all failure mode indicators (`twf`, `hdf`, `pwf`, `osf`, `rnf`) are excluded from model inputs.
- **Domain-informed derived features** ([`src/pdm/features.py`](src/pdm/features.py)):
  - `temp_diff_k = process_temperature_k - air_temperature_k` (heat dissipation failure signal)
  - `power_w = torque_nm * rotational_speed_rpm * (2 * pi / 60)` (mechanical power in Watts)
  - `strain = tool_wear_min * torque_nm` (overstrain proxy)
- **Split:** Stratified 60% train / 20% validation / 20% test (`seed=42`).

Full exploratory charts, correlation heatmaps, mutual information, feature ablation, and permutation importance are in [`notebook.ipynb`](notebook.ipynb).

---

## 3. Model Comparison & Results

Four model families were tuned with `RandomizedSearchCV` over 5-fold stratified CV:

| Model | CV PR-AUC | Val PR-AUC | Val ROC-AUC | Fit Time (s) |
|---|---:|---:|---:|---:|
| **Random Forest** *(winner)* | **0.9097** | **0.8630** | 0.9771 | 75.53 |
| XGBoost (`xgboost-cpu`) | 0.8983 | 0.8457 | 0.9829 | 10.03 |
| LightGBM | 0.8948 | 0.8603 | 0.9823 | 13.85 |
| Logistic Regression *(baseline)* | 0.4995 | 0.4692 | 0.9079 | 4.70 |

### Held-Out Test Set Performance

- **Binary (`machine_failure`, `random_forest` @ `threshold = 0.14`):**
  - **PR-AUC:** `0.8937` (vs. Logistic Regression baseline `0.4603`)
  - **ROC-AUC:** `0.9814`
  - **Recall:** `0.8529` | **Precision:** `0.7250` | **F1:** `0.7838`
  - **Expected Cost (FN=10, FP=1):** `122.0`
- **Multi-Label Failure Modes (`MultiOutputClassifier(random_forest)`):**
  - **Macro PR-AUC:** `0.7667` (vs. mean prevalence `0.0085`)
  - `hdf`: PR-AUC `0.9809`, Recall `0.8621`
  - `pwf`: PR-AUC `1.0000`, Recall `1.0000`
  - `osf`: PR-AUC `1.0000`, Recall `1.0000`
  - `twf`: PR-AUC `0.0860`, Recall `0.0000` (see Limitations)

---

## 4. Architecture & Project Structure

```text
├── data/                   # Committed AI4I 2020 CSV + data dictionary
├── models/                 # Trained bundle (model.bin) & metrics.json
├── notebook.ipynb          # Reproducible EDA, tuning & evaluation notebook
├── predict.py              # FastAPI service (/health, /predict, /predict/batch)
├── predict_test.py         # Smoke client (nominal vs. high-risk readings)
├── scripts/
│   ├── check_metrics.py    # Automated quality gate
│   └── download_data.py    # UCI fallback downloader
├── src/pdm/                # Core package (config, data, features, pipeline, evaluate, model_io, training, schemas)
├── tests/                  # Unit, API, and integration tests (>= 90% coverage)
├── train.py                # CLI training entrypoint
├── Dockerfile              # Multi-stage non-root image (< 600 MB)
├── render.yaml             # Render Blueprint configuration
└── Makefile                # Common automation targets
```

---

## 5. Setup & Installation (`uv` + Python 3.13)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed, then sync the locked environment:

### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --frozen
```

---

## 6. Training & Quality Gate

```bash
uv run python train.py
uv run python scripts/check_metrics.py
```

Or via `Makefile`:
```bash
make train
make gate
```

---

## 7. Running the API Service Locally

Start the FastAPI server (Swagger UI available at `http://localhost:9696/docs`):
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

Test with the smoke script:
```bash
uv run python predict_test.py --url http://localhost:9696
```

Or with `curl`:
```bash
curl -s http://localhost:9696/health

curl -s -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "L",
    "air_temperature_k": 302.5,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1350,
    "torque_nm": 65.0,
    "tool_wear_min": 215
  }'
```

---

## 8. Containerization (Docker)

Build and run the multi-stage non-root container:
```bash
docker build -t pdm-service .
docker run --rm -p 9696:9696 -e PORT=9696 pdm-service
```

Run the full container smoke test:
```bash
make docker-smoke
```

---

## 9. Cloud Deployment (Render)

This repository includes [`render.yaml`](render.yaml) for one-click deployment on [Render](https://render.com) (`New -> Blueprint -> Select repository`).

- **Live URL:** https://pdm-ai4i2020.onrender.com
- **Health check:** [`GET /health`](https://pdm-ai4i2020.onrender.com/health)
- **Docs (Swagger UI):** [`GET /docs`](https://pdm-ai4i2020.onrender.com/docs)

Verified against the live deployment:
```bash
uv run python predict_test.py --url https://pdm-ai4i2020.onrender.com
```

---

## 10. Quality & Testing Harness

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov=src --cov=predict --cov-report=term-missing
```

CI (`.github/workflows/ci.yml`) runs linting, type-checking, `pytest` (>= 90% coverage), fast training + quality gate, `papermill` notebook execution, Docker image size check (< 600 MB), and container smoke tests on every push and pull request.

---

## 11. Limitations & Future Work

- **TWF (Tool Wear Failure) scarcity:** With only 0.50% prevalence in the test set and stochastic tool-wear thresholds in the AI4I 2020 synthetic generator, `twf` recall is low (`0.0000`) without dedicated oversampling or mode-specific threshold tuning.
- **Synthetic dataset:** AI4I 2020 is a synthetic benchmark; real-world deployment requires calibration on actual plant telemetry and drift monitoring.

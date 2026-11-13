.PHONY: setup lint format typecheck test train gate notebook serve docker-build docker-run docker-smoke all

IMAGE_NAME ?= pdm-service
PORT ?= 9696

setup:
	uv sync --frozen

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

train:
	uv run python train.py

gate:
	uv run python scripts/check_metrics.py

notebook:
	uv run papermill notebook.ipynb /tmp/notebook_out.ipynb -p FAST 1

serve:
	uv run uvicorn predict:app --host 0.0.0.0 --port $(PORT)

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(PORT):$(PORT) -e PORT=$(PORT) $(IMAGE_NAME)

docker-smoke:
	docker run -d --name pdm_smoke -p $(PORT):$(PORT) -e PORT=$(PORT) $(IMAGE_NAME)
	@trap 'docker rm -f pdm_smoke >/dev/null 2>&1 || true' EXIT; \
	for i in $$(seq 1 30); do \
		if curl -fsS http://localhost:$(PORT)/health >/dev/null; then break; fi; \
		sleep 1; \
	done; \
	curl -fsS http://localhost:$(PORT)/health; \
	uv run python predict_test.py --url http://localhost:$(PORT)

all: lint test train gate

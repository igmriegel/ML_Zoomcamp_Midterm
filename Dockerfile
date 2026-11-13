FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev


FROM python:3.13-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY predict.py ./
COPY models/model.bin ./models/model.bin

ENV PATH="/app/.venv/bin:$PATH" \
    PORT=9696 \
    MODEL_PATH=/app/models/model.bin

USER app
EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT}"]

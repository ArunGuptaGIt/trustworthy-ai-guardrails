FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=0 \
    UV_COMPILE_BYTECODE=1

COPY --from=uv_bin /uv /uvx /bin/

COPY pyproject.toml /app/

RUN uv venv /app/.venv && \
    uv pip install --no-cache -e ".[ui]"

FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd -g 10001 appgroup && \
    useradd -u 10000 -g appgroup -s /bin/bash -m appuser

COPY --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appgroup . /app/

RUN chmod +x /app/docker-entrypoint.sh

USER appuser

EXPOSE 8000 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]

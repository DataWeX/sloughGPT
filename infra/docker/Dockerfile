FROM python:3.12-slim AS base

LABEL maintainer="SloughGPT"
LABEL description="SloughGPT API Server"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MAN_ENV="production" \
    MAN_LOG_LEVEL="INFO" \
    MAN_HOST="0.0.0.0" \
    MAN_PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY apps/api/server/requirements.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt \
    && pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    sentence-transformers \
    psutil

COPY apps/api/server /app/apps/api/server
COPY packages/core-py /app/packages/core-py
COPY packages/sdk-py /app/packages/sdk-py
COPY pyproject.toml README.md LICENSE /app/

RUN mkdir -p /app/data /app/logs /app/cache /app/models /app/datasets

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "/app/apps/api/server/main.py"]

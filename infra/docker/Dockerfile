FROM python:3.12-slim AS builder

LABEL maintainer="SloughGPT"
LABEL description="SloughGPT API Server"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY apps/api/server/requirements.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt \
    && pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir psutil

COPY apps/api/server /build/apps/api/server
COPY packages/core-py /build/packages/core-py
COPY packages/sdk-py /build/packages/sdk-py
COPY pyproject.toml README.md LICENSE /build/

RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -e packages/core-py \
    && pip install --no-cache-dir -e packages/sdk-py/sloughgpt_sdk

# ─── Runtime stage (no build tools) ─────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SLO_ENV="production" \
    SLO_LOG_LEVEL="INFO" \
    SLO_HOST="0.0.0.0" \
    SLO_PORT=8000

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY apps/api/server /app/apps/api/server
COPY packages/core-py /app/packages/core-py
COPY packages/sdk-py /app/packages/sdk-py
COPY pyproject.toml README.md LICENSE /app/

RUN mkdir -p /app/data /app/logs /app/cache /app/models /app/datasets

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "/app/apps/api/server/main.py"]

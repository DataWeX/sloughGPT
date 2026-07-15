# SloughGPT Environment Variables

Complete reference for all environment variables used in SloughGPT.

## Quick Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SLO_API_KEY` | Yes | - | API key for authentication |
| `SLO_JWT_SECRET` | Yes | - | Secret for JWT token signing |
| `SLO_ENV` | No | `development` | Environment mode |
| `SLO_HOST` | No | `0.0.0.0` | Server host |
| `SLO_PORT` | No | `8000` | Server port |
| `SLO_RELOAD` | No | `false` | Enable auto-reload on file changes |

**Legacy names:** older docs and images used a typo (`SLAUGHGPT_*`). The server still accepts `SLAUGHGPT_API_KEY`, `SLAUGHGPT_JWT_SECRET`, and `SLAUGHGPT_API_KEYS` if the `MAN_*` counterparts are unset. Prefer `MAN_*` for new deployments.

---

## Authentication

### SLO_API_KEY
**Required in production**

API key for authenticating requests.

```bash
# Generate a secure key
openssl rand -hex 32

# Set in .env
SLO_API_KEY=your-generated-key-here
```

### SLO_API_KEYS
**Optional**

Comma-separated list of multiple valid API keys.

```bash
SLO_API_KEYS=key1,key2,key3
```

### SLO_JWT_SECRET
**Required in production**

Secret key for signing JWT tokens.

```bash
# Generate a secure secret
openssl rand -hex 64

# Set in .env
SLO_JWT_SECRET=your-64-character-secret
```

---

## Server Configuration

### SLO_ENV
**Optional**

Environment mode.

```bash
SLO_ENV=development  # or production
```

### SLO_HOST
**Optional**

Server bind address.

```bash
SLO_HOST=0.0.0.0  # Default
```

### SLO_PORT
**Optional**

Server port.

```bash
SLO_PORT=8000  # Default
```

### SLO_RELOAD
**Optional**

Enable uvicorn auto-reload on Python file changes.

```bash
SLO_RELOAD=true  # Default: false
```

---

## Logging

### SLO_LOG_LEVEL
**Optional**

Minimum log level for server output.

```bash
SLO_LOG_LEVEL=INFO   # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### SLO_LOG_FORMAT
**Optional**

Output format for logs. Use `json` for structured logging (log aggregation, ELK stack, Datadog).

```bash
SLO_LOG_FORMAT=human  # "human" (colored terminal) or "json" (structured JSON lines)
```

JSON output example:
```json
{"ts":"2026-07-15T03:33:03.454Z","level":"INFO","logger":"man.api","msg":"Server started","tag":"START","ctx":{"port":8000}}
```

### Type Tags

Every log line includes a category tag for quick visual scanning:

| Tag | Category | Example |
|-----|----------|---------|
| `[START]` | Server startup, config | `Phase 4: loading model` |
| `[MODEL]` | Model loading, providers | `Registered hf-default: Qwen2.5-0.5B` |
| `[SOUL]` | Soul management, personality | `Found 7 souls` |
| `[REQ]` | HTTP requests, timing | `GET /chat 200 (0.34s)` |
| `[INF]` | Inference, streaming, knowledge | `Client disconnected` |
| `[TRAIN]` | Training pipeline | `Auto-train configured` |
| `[INFRA]` | Infrastructure, deployment | `RateLimitMiddleware registered` |
| `[AUTH]` | Authentication | `Token expiring` |

### Error Codes

Structured error codes for programmatic handling:

| Code | Category | Description |
|------|----------|-------------|
| `E_AUTH_MISSING` | Auth | No API key provided |
| `E_AUTH_EXPIRED` | Auth | Token expired |
| `E_AUTH_INVALID` | Auth | Invalid token |
| `E_AUTH_FORBIDDEN` | Auth | Insufficient permissions |
| `E_MODEL_LOAD` | Model | Failed to load model |
| `E_MODEL_OOM` | Model | Out of memory |
| `E_MODEL_TIMEOUT` | Model | Inference timeout |
| `E_MODEL_CRASH` | Model | Model crashed |
| `E_MODEL_WARMUP` | Model | Warmup failed |
| `E_INF_TOKENIZER` | Inference | Tokenizer error |
| `E_INF_GENERATION` | Inference | Generation failed |
| `E_INF_CACHE` | Inference | Cache error |
| `E_INFRA_STARTUP` | Infra | Startup failed |
| `E_INFRA_TIMEOUT` | Infra | Request timeout |
| `E_INFRA_REGISTRY` | Infra | Registry error |
| `E_INFRA_PROVIDER` | Infra | Provider error |
| `E_VAL_REQUEST` | Validation | Invalid request |
| `E_VAL_FIELD` | Validation | Invalid field |
| `E_TRAIN_DATA` | Training | Data error |
| `E_TRAIN_CRASH` | Training | Training crashed |
| `E_TRAIN_CHECKPOINT` | Training | Checkpoint error |
| `E_DOMAIN` | Domain | Business logic error |

---

## Legacy / Deprecated

The following environment variable names are accepted as fallbacks:

| Legacy Name | Modern Name |
|-------------|-------------|
| `SLAUGHGPT_API_KEY` | `SLO_API_KEY` |
| `SLAUGHGPT_JWT_SECRET` | `SLO_JWT_SECRET` |
| `SLAUGHGPT_API_KEYS` | `SLO_API_KEYS` |

These are read from `settings.py` if the `MAN_*` variant is unset. New deployments should use only `MAN_*` names.

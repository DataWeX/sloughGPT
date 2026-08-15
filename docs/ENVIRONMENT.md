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

**Legacy names:** older docs and images used a typo (`SLAUGHGPT_*`). The server still accepts `SLAUGHGPT_API_KEY`, `SLAUGHGPT_JWT_SECRET`, and `SLAUGHGPT_API_KEYS` if the `SLO_*` counterparts are unset. Prefer `SLO_*` for new deployments.

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

### SLO_ENABLE_PROCESS_GUARD
**Optional**

Run model inference in a guarded subprocess for crash isolation. When enabled,
the model is served from a worker process so a crash (OOM, segfault) cannot
take down the API server; the guard auto-restarts the worker up to
`SLO_GUARD_MAX_RESTARTS` times.

Can also be toggled at runtime via `POST /models/process-guard` (`{"enabled": true/false}`).

```bash
SLO_ENABLE_PROCESS_GUARD=true  # Default: true
```

Related knobs: `SLO_GUARD_MAX_RESTARTS` (default `3`), `SLO_GUARD_RESTART_DELAY`
(seconds), `SLO_GUARD_MEMORY_LIMIT_MB`, `SLO_PROCESS_GUARD_CONCURRENT`.

---

## Memory (Auto-Memory)

### Behavior
Every completed chat turn (user + assistant, combined ≥ `SLO_MEMORY_MIN_CHARS`
chars) is distilled into knowledge facts automatically. When facts exist, the
chat loop builds a context frame: the RAG layer pulls relevance-gated facts
(`MIN_RELEVANCE_SCORE` + topical overlap, via `enrich_with_knowledge`) and the
memory layer pulls prior-session episodes; both are injected into the model
prompt alongside the context-manager system prompt. Disable entirely with
`SLO_MEMORY_ENABLED=false`.

### SLO_MEMORY_ENABLED
**Optional** — Default: `true`

Master switch for the auto-memory layer. When `false`, every memory method
(`remember`, `retrieve`, `store`) no-ops so the chat loop and future task
executor are completely unaffected.

```bash
SLO_MEMORY_ENABLED=false   # disable long-term learning
```

### SLO_MEMORY_MIN_CHARS
**Optional** — Default: `80`

Minimum combined length (user message + assistant response) before a completed
turn is worth remembering. Short small-talk turns are noise, not knowledge.

```bash
SLO_MEMORY_MIN_CHARS=120   # only remember substantive exchanges
```

### SLO_MEMORY_MAX_FACTS
**Optional** — Default: `5`

Maximum number of memory facts returned by a single retrieval.

```bash
SLO_MEMORY_MAX_FACTS=10
```

### SLO_MEMORY_STORE_PATH
**Optional** — Default: `data/memory`

Directory used by the task-backed memory store (`memory.remember` /
`memory.store` / `memory.consolidate` tasks). Every successfully processed
task appends one JSONL provenance record to `<store_path>/facts.jsonl`. The
learner's `KnowledgeMemoryProvider` remains the retrieval index; this archive
is the durable, inspectable record of task-mined facts.

### SLO_MEMORY_SYNC
**Optional** — Default: `false`

When `true`, `remember()` stores inline instead of being offloaded to a worker
thread. Useful for tests and CLI/task producers; the chat loop always uses
`asyncio.to_thread`.

### SLO_MEMORY_CONSOLIDATION_THRESHOLD
**Optional** — Default: `0.80`

Minimum n-gram cosine similarity for two same-topic facts to be treated as
near-duplicates by the `memory.consolidate` task. When merged, the longest
fact is kept and the duplicates are deleted. The default `0.80` collapses
near-verbatim copies (measured ~0.845) while keeping genuine paraphrases
(~0.586) and cross-topic facts distinct. Lower to merge more aggressively
(paraphrases), raise to only collapse near-verbatim copies.

```bash
SLO_MEMORY_CONSOLIDATION_THRESHOLD=0.90   # conservative: only near-verbatim
```

### SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES
**Optional** — Default: `60`

How often the server runs an automatic maintenance pass. Each pass prunes
the provenance archive to `SLO_MEMORY_ARCHIVE_RETENTION_DAYS` and enqueues
one `memory.consolidate` task, so near-duplicate facts are merged and the
audit trail stays bounded without operator action. Set to `0` to disable
automatic maintenance (consolidation can still be run manually via
`sloughgpt memory consolidate`).

```bash
SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES=1440   # once a day
```

### SLO_MEMORY_ARCHIVE_RETENTION_DAYS
**Optional** — Default: `30`

Every `memory.store`, `memory.remember`, and `memory.consolidate` task append
a durable record to the provenance archive (`<store_path>/facts.jsonl`). The
archive is bounded automatically: every maintenance pass (see
`SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES`) prunes records older than this
window, and it can be pruned on demand with
`sloughgpt memory archive --prune-days N` (default retention `30`). Records
without a timestamp are treated as oldest and pruned first. The archive is
append-only and fail-closed: a corrupt line is skipped on read, and a failed
prune rewrite leaves the original file untouched.

```bash
SLO_MEMORY_ARCHIVE_RETENTION_DAYS=90   # keep three months of provenance
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

### SLO_LOG_COLOR
**Optional**

Force or disable ANSI color output in `human` format. Defaults to auto-detection (colors are emitted only when stderr is a TTY and `NO_COLOR` is unset). `NO_COLOR=1` always disables colors.

```bash
SLO_LOG_COLOR=1   # force colors even when stderr is piped/redirected
SLO_LOG_COLOR=0   # force plain text even in a terminal
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
| `[INFO]` | Per-request generation telemetry (debug) | `generate_sync (mode=guard)` |
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

These are read from `settings.py` if the `SLO_*` variant is unset. New deployments should use only `SLO_*` names.

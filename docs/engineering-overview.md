# SloughGPT — Engineering Overview

Self-hosted LLM platform: train from scratch, serve, chat. One core engine, four client surfaces.

---

## System Design

```
                    ┌──────────────────────────────────┐
                    │            Clients               │
                    │                                  │
                    │  ┌────────┐  ┌────────┐         │
                    │  │  Web   │  │ Mobile │         │
                    │  │ Next.js│  │ RN/Expo│         │
                    │  └───┬────┘  └───┬────┘         │
                    │      │           │               │
                    │  ┌───┴────┐  ┌───┴────┐         │
                    │  │  CLI   │  │Gateway │         │
                    │  │ Python │  │  Rust  │         │
                    │  └───┬────┘  └───┬────┘         │
                    └──────┼───────────┼───────────────┘
                           │   HTTP    │
                           ▼           ▼
                    ┌──────────────────────────────┐
                    │         API Server           │
                    │      FastAPI + Uvicorn       │
                    │        :8000                 │
                    │   48 routers, background     │
                    │   daemons, error taxonomy    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Core Logic            │
                    │   domain   │
                    │                              │
                    │  training/  inference/        │
                    │  models/    cognitive/        │
                    │  soul/      shell/            │
                    │  infrastructure/              │
                    └──────────────────────────────┘
```

**Principle:** Core logic is framework-agnostic. No FastAPI, React, or CLI imports in `domain/`. This keeps the engine portable across API server, CLI local mode, notebooks, and workers.

---

## The Five Components

### 1. Core Engine (`domain/`)

29 domain modules. The brain.

| Layer              | Modules           | What It Does                                                                         |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------ |
| **Training**       | `training/`       | Character-level trainer, GPT-2 distillation, LoRA adapters, feedback-driven training |
| **Inference**      | `inference/`      | KV cache, vector store, SLNC memory-mapped format, context management                |
| **Models**         | `models/`         | SloughGPTModel (RoPE + SwiGLU + RMSNorm), SloTransformer, arch detection             |
| **Cognitive**      | `cognitive/`      | Memory, reasoning, learning, knowledge graph                                         |
| **Infrastructure** | `infrastructure/` | Error taxonomy, EventBus, lifecycle, task queue, rate limiter, config                |

**Key decisions:**

- Pure NumPy autograd. No PyTorch dependency for core inference/training.
- `.slnc` memory-mapped format — 2.2x faster load, demand paging.
- `.sou` checkpoint format — 1960x faster than JSON.
- AVX2 int8/int4 GEMM for CPU quantized inference.

---

### 2. API Server (`apps/api/server/`)

FastAPI backend. The hub everything connects to.

**48 routers** covering: chat, inference, training, models, datasets, knowledge, agents, souls, feedback, shell, VM, multimodal, voice, images, health, config, auth, security, metrics.

**Startup sequence:**

1. Path bootstrapping → logging → CORS
2. `StartupOrchestrator` runs 7 lifecycle hooks (task queue, config, model load, registry, routers, daemons)
3. Background daemons: feedback workflow, health monitor, watchdog, auto-trainer, RAG ingestion

**Request flow:**

```
Request → CorrelationId → ReadinessGate → RateLimit → Timeout → Router → Handler → Response
                                                   ↓ (error)
                                            classify_and_raise()
                                                   ↓
                                            AppError → exception_handler → JSON
```

**Error handling:** 30 `E_UPPER_CASE` codes, `AppError` hierarchy, single EventBus emission point in `_app_error_handler`. Routers use `classify_and_raise()` — classify only, no emit.

**State:** Thread-safe via `AtomicRef` in `state.py`. Model state, training state, config state.

---

### 3. Web App (`apps/web/`)

Next.js 16 frontend. 59 routes.

**Data flow:**

```
Component → Controller (lib/*-controller.ts) → http-client.ts → API Server
                                                           ↓
Component ← Zustand Store ← Controller ← Response
```

**`http-client.ts`** is the single API layer. Interceptors, cache, circuit breaker, dedup, throttling. Never raw `fetch()`.

**40+ controllers** — one per domain (chat, model, training, dataset, knowledge, etc.). Controllers call http-client, transform data, update stores.

**88 components** using `@sloughgpt/strui` library. Noir Violet design system, dark/light themes.

**Testing:** 3048+ Vitest tests, 6 Cypress E2E specs.

---

### 4. Mobile App (`apps/mobile/`)

React Native 0.86. 32 screens, 32 services.

**Two inference paths:**

- **Remote:** API server via `api-client.ts` (same as web)
- **On-device:** `llama-rn` (Metal, 15-30 tok/s) or JS SloNet (pure JS, 2-5 ms/token)

**Offline mode:** `offline-cache.ts` queues messages when offline, flushes when online.

**Navigation:** React Navigation. 5 tabs (Home, Chat, Models, Tools, Settings). Deep linking via `sloughgpt://`.

**Training data collection:** `training-collector.ts` captures chat pairs for incremental LoRA training.

---

### 5. Gateway (`apps/gateway/`)

Rust/Axum reverse proxy (`slough-gateway`) — **the** edge, bound to
`:8080`. No proxy in front of it: the gateway itself owns the public
socket, CORS, and every transport policy. (`scripts/edge-proxy.mjs` is an
optional harness socket only — not part of the serving topology.)

**Three hops, each with one job:**

```
browser ─► slough-gateway :8080 ─► FastAPI :8000 ─► domain/ + infrastructure/
           CORS, filters, health,   routers, envelope,
           compression              error taxonomy
```

**Edge owns compression — one encoder, one hop, no double-encode.** The relay
asks the sidecar for `accept-encoding: identity` and re-encodes per _client_
negotiation (zstd > gzip; streaming — the body is never buffered; identity for
`Range`/206/`no-transform`/tiny known bodies/HEAD/non-negotiating clients).
SSE token streams compress from byte 0: no `Content-Length`, so no size gate
and no first-token hold. Benchmark A/B = omit vs send `Accept-Encoding`;
ratios live in `scripts/benchmark_gateway_compression.py --class sse`.

**Contract pass-through:** only the 8 RFC 9110 hop-by-hop headers are stripped —
`Authorization`, `X-Correlation-ID`, `Retry-After`, and the standard envelope
cross untouched. CORS is allow-any (bearer-token auth, no cookies), matching
`http-client.ts`.

**Health is a contract passthrough:** `/health` and `/health/detailed` relay
FastAPI's bodies verbatim — the frontend reads `model_loaded`/`model_type`
and `path_latencies`/`health_score` from them — with `gateway: "rust"`
injected only when the body is a plain JSON object. The edge envelope
(`status: degraded` + `sidecar`) is the unreachable-core fallback.

**Role:** Fast entry point. Generic byte-relay to Python core. Strict path filters (traversal → 403). Opt-in `MAN_GATEWAY_DENY` prefixes + `MAN_GATEWAY_CHAT_ONLY=1`. Background health checker (3s poll, parses API `data` envelope). Serves static files when present. **Self-supervising** (`MAN_GATEWAY_SUPERVISE=1`, default in `host_gateway.sh`): a parent loop in the same binary respawns the worker on crash (1s→30s backoff, resets after 60s healthy); SIGTERM chain-stops; a killed parent leaves the orphan worker serving — no systemd, no OS coupling. **Rate limiting** is a verbatim port of Python's `RateLimitMiddleware`: sliding windows, the GPU-heavy route table (`/chat/stream` 10/60 …), per-client global (`MAN_GATEWAY_RATE_LIMIT`/`MAN_GATEWAY_RATE_WINDOW`, default 300/60) with localhost ×10, best-effort workspace-JWT limits, `X-RateLimit-*` headers, `429 {"detail": …}` — health probes, CORS preflights (CorsLayer is outer), and `/static` bypass. `MAN_GATEWAY_MAX_STREAMS` (default 32, `0` off) additionally caps concurrent SSE streams, held until each body finishes — as an **AIMD controller**: the cap halves when ≥12.5% of streams in a 2s window fail against the core (transport errors, ≥500 relays) and grows +1 per healthy window (floor 1, visible as `gateway.stream_limit`), so a struggling core sheds early while a healthy one returns to the ceiling; the breaker still handles the dead-core case.

**Performance control plane.** The edge is the only place that sees *all* traffic (including what it sheds), so it owns two observations FastAPI structurally cannot have: **stats** (`GET /gateway/stats`, exempt from rate limits and the breaker) — per-route requests/2xx/4xx/5xx, shed counters (`429`/`503`), upstream errors, identity-vs-wire byte totals (compression saved), log2-µs latency histograms with p50/p95, an in-flight SSE gauge, and a **per-client cut** (peer IP, same 32-slot + `(other)` cap, sorted by activity — edge-internal health/stats polls excluded as overhead); and a **circuit breaker** — after N consecutive edge-observed ≥500/transport failures (`MAN_GATEWAY_BREAKER_FAILURES`, default 5, `0` off), the breaker opens and sheds instantly with `503 {"detail":"Upstream unavailable."}` + `Retry-After` for `MAN_GATEWAY_BREAKER_OPEN` seconds (default 5), then admits exactly one half-open probe; a late success while open never masks a failing run, and health-poll results never feed it (masking risk). Complementing the counters, a **JSONL access log** (`MAN_GATEWAY_ACCESS_LOG`, default `logs/gateway-access.jsonl`, `0` off) writes one flushed line per edge-observed request — timestamp, latency, method, path, route, status, shed reason, client — including exempt health/stats probes, so "what consumes the edge, and when" survives restarts.

**Not a logic layer.** No auth, no error transformation. Rate limiting is the one policy the edge *enforces first* rather than understands — windows are ported verbatim from `apps/api/server/infrastructure/rate_limit_middleware.py`, which still runs behind the gateway but degrades to a no-op (removal pending approval). Otherwise just routing + filters + compression + health + static files.

**Catch-all:** Any allowed route not explicitly defined is proxied to Python core.

---

## Cross-Cutting Concerns

### Error Handling

```
domain/infrastructure/_internal/errors.py    — ErrorCode enum, ERROR_REGISTRY, AppError
apps/api/server/schemas/common.py   — raise_error(), classify_and_raise()
apps/api/server/infrastructure/     — exception_handlers.py (single emission point)
```

**Flow:** Router catches → `classify_and_raise()` classifies + raises → exception handler emits EventBus + returns JSON.

**30 codes:** `E_INTERNAL`, `E_NOT_FOUND`, `E_MODEL_OOM`, `E_AUTH_MISSING`, `E_TIMEOUT`, etc.

### EventBus

Pub/sub singleton. `emit()` / `on()` / `once()`. Fire-and-forget. Used for cross-module communication (training completed, error raised, model loaded).

### Lifecycle

Ordered startup/shutdown. Services register hooks with dependencies. `StartupOrchestrator` runs them in topological order.

### Task Queue

Async priority queue with worker pool. Pause/resume/cancel. Retries. SSE events. Foundation for training, feedback, RAG ingestion.

---

## Data Flow Examples

### Chat

```
User types message
  → Web/Mobile/CLI sends POST /chat
  → API server routes to inference engine
  → Model generates tokens (KV cache)
  → SSE stream back to client
  → Client renders incrementally
```

### Training

```
User starts training
  → POST /training/start
  → TaskQueue submits training job
  → Trainer runs in background thread
  → SSE stream emits progress (loss, epoch, step)
  → Checkpoint saved on completion
  → EventBus emits training.completed
  → Feedback workflow may trigger LoRA update
```

### Model Loading

```
POST /models/load
  → StartupOrchestrator loads model in background
  → SLNC parser reads memory-mapped weights
  → Quantization applied (AVX2 int8/int4)
  → Model state: IDLE → LOADING → READY
  → ReadinessGate allows inference requests
```

---

## Testing Strategy

| Component  | Framework   | Count    | Focus                                       |
| ---------- | ----------- | -------- | ------------------------------------------- |
| Core Logic | pytest      | 399+     | Training convergence, inference correctness |
| API Server | pytest      | 100+     | Endpoint contracts, error handling          |
| Web        | Vitest      | 3048+    | Components, controllers, hooks              |
| Web E2E    | Cypress     | 6 specs  | Critical user flows                         |
| CLI        | pytest      | 16 files | Command parsing, output formatting          |
| Mobile     | Jest + RNTL | 25+      | Screens, services                           |

---

## Deployment

```bash
# Docker (all services)
docker-compose up -d

# Local dev
python -m apps.api.server.main --reload    # API :8000
cd apps/web && npm run dev                  # Web :3001
cd apps/gateway && cargo run                # Gateway :8080 (MAN_GATEWAY_PORT to move)

# Or one shot (API + Web; opt-in edge):
./scripts/dev-stack.sh
MAN_DEV_GATEWAY=1 ./scripts/dev-stack.sh    # + gateway :8080 → API

# CLI
pip install -e .
sloughgpt serve
sloughgpt chat
sloughgpt shell
```

---

## What's Done, What's Next

| Phase                     | Status |
| ------------------------- | ------ |
| Core inference engine     | Done   |
| Training pipeline         | Done   |
| Model serving + health    | Done   |
| API + CLI                 | Done   |
| Web frontend              | Done   |
| Mobile app                | Done   |
| Gateway                   | Done   |
| Multi-user / multi-tenant | Done   |
| **Cloud/edge deployment** | Next   |
| **Enterprise features**   | Next   |

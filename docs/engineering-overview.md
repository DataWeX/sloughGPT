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
                    │   packages/core-py/domains   │
                    │                              │
                    │  training/  inference/        │
                    │  models/    cognitive/        │
                    │  soul/      shell/            │
                    │  infrastructure/              │
                    └──────────────────────────────┘
```

**Principle:** Core logic is framework-agnostic. No FastAPI, React, or CLI imports in `domains/`. This keeps the engine portable across API server, CLI local mode, notebooks, and workers.

---

## The Five Components

### 1. Core Engine (`packages/core-py/domains/`)

29 domain modules. The brain.

| Layer | Modules | What It Does |
|-------|---------|--------------|
| **Training** | `training/` | Character-level trainer, GPT-2 distillation, LoRA adapters, feedback-driven training |
| **Inference** | `inference/` | KV cache, vector store, SLNC memory-mapped format, context management |
| **Models** | `models/` | SloughGPTModel (RoPE + SwiGLU + RMSNorm), SloTransformer, arch detection |
| **Cognitive** | `cognitive/` | Memory, reasoning, learning, knowledge graph |
| **Infrastructure** | `infrastructure/` | Error taxonomy, EventBus, lifecycle, task queue, rate limiter, config |

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

Rust/Axum reverse proxy. 574 lines.

**Role:** Fast entry point. Routes requests to Python core. Serves static files (Next.js build). Background health checker (3s poll).

**Not a logic layer.** No auth, no rate limiting, no error transformation. Just routing + health + static files.

**Catch-all:** Any route not explicitly defined is proxied to Python core.

---

## Cross-Cutting Concerns

### Error Handling

```
domains/infrastructure/errors.py    — ErrorCode enum, ERROR_REGISTRY, AppError
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

| Component | Framework | Count | Focus |
|-----------|-----------|-------|-------|
| Core Logic | pytest | 399+ | Training convergence, inference correctness |
| API Server | pytest | 100+ | Endpoint contracts, error handling |
| Web | Vitest | 3048+ | Components, controllers, hooks |
| Web E2E | Cypress | 6 specs | Critical user flows |
| CLI | pytest | 16 files | Command parsing, output formatting |
| Mobile | Jest + RNTL | 25+ | Screens, services |

---

## Deployment

```bash
# Docker (all services)
docker-compose up -d

# Local dev
python -m apps.api.server.main --reload    # API :8000
cd apps/web && npm run dev                  # Web :3001
cd apps/gateway && cargo run                # Gateway :8080

# CLI
pip install -e .
sloughgpt serve
sloughgpt chat
sloughgpt shell
```

---

## What's Done, What's Next

| Phase | Status |
|-------|--------|
| Core inference engine | Done |
| Training pipeline | Done |
| Model serving + health | Done |
| API + CLI | Done |
| Web frontend | Done |
| Mobile app | Done |
| Gateway | Done |
| **Multi-user / multi-tenant** | Next |
| **Cloud/edge deployment** | Next |
| **Enterprise features** | Next |

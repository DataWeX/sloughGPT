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
                    │  │  Vite  │  │ RN/Expo│         │
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

Vite + React frontend. 59 routes.

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

Rust/Axum reverse proxy (`slough-gateway`, binds `:8080`).

**Role:** Fast entry point. Generic byte-relay to Python core. Stateless zstd/gzip compression (identity fallback for SSE/Range/small/HEAD). Strict path filters (traversal → 403). Opt-in `MAN_GATEWAY_DENY` prefixes + `MAN_GATEWAY_CHAT_ONLY=1`. Background health checker (3s poll, parses API `data` envelope). Serves the **static site document root** when a build exists (`MAN_STATIC_DIR`, default `apps/web/dist-vite`, then `apps/web/.next/static`): file hit → asset, browser navigation (`Accept: text/html`) → `index.html` for SPA routes, data requests → relay. Routing is by **request kind, not path prefix** — the SPA and the API own the same top-level names (`/training`, `/models`, `/chat`), and the only browser _navigation_ onto API surface is `/docs`, `/redoc`, `/openapi.json` (path contract). Result: shell + API + SSE on one origin, and the shell still loads when the sidecar is down.

**Not a logic layer.** No auth, no rate limiting, no error transformation. Just routing + filters + compression + health + static files.

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

### Prompt Assembly (layer contract)

One funnel, many writers: **`stream_turn`** (`apps/api/server/routers/inference.py`) is the single composition point for every prompt the model sees. This subsection is the declared map — writers, placement, precedence, budgets, and the gate that proves each layer. Audited 2026-11-24.

```
TurnRequest (wire fields)
  → _resolve_turn_scope      AppScope: conversation delta over global settings,
                            once at the seam, before any transport read
  → system head              wire system_prompt → + scope system_prompt (additive)
                            → + agent instructions → + trait steering (system_extra)
  → volatile tail            _append_volatile_context: knowledge + frame + RAG
                            (+ consciousness reflection when use_consciousness)
  → trim                     _trim_messages_to_budget(max_context_length, max_tokens)
  → provider chat_stream     chat template renders; meta weights ride as generation
                            params, never as prompt text
```

| Layer                | Writer                                        | Placement              | Precedence                                       | Budget / kill-switch                                              | Gate                                                                       |
| -------------------- | --------------------------------------------- | ---------------------- | ------------------------------------------------ | ------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| Wire `system_prompt` | `TurnRequest`                                 | system head            | base layer                                       | —                                                                  | sequence test (below)                                                      |
| AppScope scope       | `_resolve_turn_scope` (additive after wire)   | system head            | conversation-over-global, presence-based; `scope_applied` protects meta weights | session-scoped via `PUT /chat/sessions/{id}`; corrupt delta fails safe to inheritance | `tests/core-py/test_app_scope.py` + `apps/api/server/tests/test_app_scope.py` (35) |
| Agent instructions   | `stream_turn` (`agent_id`)                    | system head, appended  | after scope                                      | agent missing → skipped, fail-open                                | —                                                                          |
| Trait steering       | ContextCore managers → `frame.system_extra`   | system head, appended  | after agent                                      | weights clamped [0,1]; fail-open (never a precondition for inference); sampler nudge dark (`SLO_META_WEIGHT_NUDGE=0`) | `tests/test_inference_generate.py`                                         |
| Consciousness reflection | `_apply_consciousness_reflection`         | volatile tail          | `use_consciousness` wire flag                    | off for bare probes (cards 20261112_140010, 20261114_092948)       | sequence test (below)                                                      |
| Knowledge + frame    | `_enrich_knowledge` + `ContextCore` frame      | volatile tail           | merged: `knowledge_retrieved + frame_context + req.knowledge` | top 5 facts; frame skipped when memory empty                      | `tests/test_rag_context_budget.py`                                         |
| RAG block            | `_build_rag_block`                            | volatile tail           | relevance-gated, `use_rag` flag                   | 2000 chars (`_RAG_CONTEXT_MAX_CHARS`)                              | `tests/test_rag_context_budget.py`                                         |
| Context contract parts | `budget_parts` (`domain/inference/_internal/context_contract.py`) | volatile tail (knowledge block) | session 1.0 > request 0.9 > memory 0.8 > rag 0.7, then score desc | 512 (`DEFAULT_KNOWLEDGE_BUDGET`), whole-part drops + honest `omitted_count` | `tests/core-py/test_context_contract.py` (26)                             |
| Chat template        | provider / tokenizer                          | wraps everything       | final render                                     | tokenizer template                                                | tokenizer tests                                                            |
| Meta weights         | `_apply_meta_weights`                         | params, not text       | explicit (`model_fields_set` ∪ `scope_applied`) beats heuristic | sampler nudge dark (`SLO_META_WEIGHT_NUDGE=0`)                   | `tests/test_inference_generate.py`                                         |

**Precedence rules.** Text precedence = placement order in the system head (wire → scope → agent → steering). Volatile context never touches the token prefix — per-turn context rides the last user turn so cross-turn KV reuse still matches (re-prefill fix, card 20261103_071231). Scope precedence is presence-based first-set-wins per field (an explicitly-set value is user intent and outranks a derived one).

**Open ruling — user decision required before any code.** Does a scoped AppScope `system_prompt` *outrank* trait steering / injected RAG, or only order them? Today it is additive-only (scope appends after the wire prompt, steering after that, retrieved context in the tail). Observed 2026-11-24: a scoped persona was ignored by the 0.5B model under RAG noise — behavioral evidence, not a gate.

**Verification — proven vs gap.** Stage *order* is pinned by `TestStreamTurnResponseSequence` / `TestStreamTurnBareProbeGates` (`tests/server/test_inference_router.py`); each layer's budget and precedence by its own suite (table). **Gap: no provider-boundary prompt-capture gate** — nothing asserts the final rendered string the model actually receives. Closing it = capture at the provider seam and assert against this map (future card).

Related: memory-layer env (`docs/ENVIRONMENT.md`), steering diagram (`docs/NAVIGATION_AND_SOULS.md` §2.8), retrieved-context contract (`domain/inference/_internal/context_contract.py`).

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
cd apps/web && npm run dev                  # Web :5173
cd apps/gateway && cargo run                # Gateway :8080

# Or one shot (API + Web; opt-in edge):
./scripts/dev-stack.sh
MAN_DEV_GATEWAY=1 ./scripts/dev-stack.sh    # + gateway :8080 → API

# CLI
pip install -e .
sloughgpt serve
sloughgpt chat
sloughgpt shell
```

### Port map (container truth, audited 2026-10-08 — card 257d310b)

The web container listens on **:3000 inside the container**
(`apps/web/Dockerfile`: nginx `listen 3000` + `/health`, `EXPOSE 3000`,
`HEALTHCHECK` — the only web image since Next.js was removed). Everything
that targets `web:3000` or `localhost:3000` _inside_ the compose network is
therefore valid.

| Context                                        | Host ports                                            | Notes                                                                                                                 |
| ---------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Live systemd dev stack                         | web **:5173** (vite), api :8000, gateway :8080        | Independent of docker; never :3000 (`e47e19ee` retired those claims)                                                  |
| `docker-compose.yml` (dev)                     | web `3000:3000`, api `8000:8000`, gateway `8080:8080` | Host `curl localhost:3000` valid **here only**                                                                        |
| `docker-compose.prod.yml`                      | nginx only: `${NGINX_HTTP_PORT:-80}:80`, `:443`       | nginx routes `/ → web:3000`, `/v1\|chat\|training\|health → api:8000`, static → gateway. Nothing on host :8000/:3000  |
| grafana / prometheus / postgres / redis (prod) | none                                                  | Internal only — grafana's `:3000/api/health` healthcheck is Grafana's own in-network default port, not a host publish |

Consequences: `infra/scripts/deploy.sh` health checks read **container
health** (`docker inspect`) instead of host curls that can never pass;
`.env.example`'s `NEXTAUTH_URL` (Next-path only — host `dev:e2e` runs
Next on `:3010`, dev-compose maps it to `:3000`; the live vite stack
ignores it) applies to the Next path (prod compose defaults to
`http://localhost`);
the web image is `apps/web/Dockerfile` (Vite static on nginx). Grafana has
no nginx route — no host URL exists for it.

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

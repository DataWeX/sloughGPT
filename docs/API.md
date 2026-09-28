# SloughGPT API & SDK Documentation

Reference for the HTTP API served by `apps/api/server` and the SDKs that call it.
The SDK surface is **real** — every method maps to a live backend endpoint. There
are no simulated, in-memory, or local-only API methods.

## Response Envelope

Two sanctioned response styles; both are enforced by
`tests/contract/test_api_envelope.py`:

**1. Envelope (default).** Most endpoints wrap responses in `StandardResponse`
via `success_response()`:

```json
{ "status": "success", "data": { ... } }
```

List-returning enveloped endpoints put the array inside `data` (e.g.
`/registry/models` → `data: { "models": [...], "count": N }`).

**2. Typed.** A route registered with `response_model=X` returns the bare typed
payload — `GET /knowledge` → `list[KnowledgeItemOut]`, `POST /shell/exec` →
`ShellExecResponse`, `POST /inference/generate` → `InferResponse`. The type is
the contract (declared in OpenAPI).

All SDK client methods unwrap the envelope automatically, so callers receive
the payload either way: enveloped responses unwrap to `data`, typed responses
pass through unchanged (`http-client.ts` unwraps iff `status` and `data` are
both present). An endpoint must not mix the two styles.

## Shared Core Contract (one system image)

The HTTP surface + process state form **one shared image** that both clients (web
`apps/web`, native `apps/mobile`) boot against in the same environment. Core logic
lives in `domain/`; routers are a thin **read-model** over it. This section is the
non-negotiable contract — a feature is "translated" between app and web by pointing
both shells at the same endpoints, never by giving one shell a private path.

### Rules

1. **Translate at the edge, never in the core.** A shell (web or mobile) may _shape_
   responses (trim fields, paginate, re-envelope into its own DTO). It must never
   reach past the endpoint layer into `domain/` singletons. `apps/api/server/routers/mobile.py`
   is the one known violation (it imports `controllers.*`, `routers.inference._instance`,
   `get_training_engine()`, etc. directly) and is being re-pointed onto shared handlers.
2. **One transport agreement for everything.** All API traffic goes through the
   `http-client.ts` family (`apiGet/apiPost/.../streamSSE`) on web and `api-client.ts`
   on mobile. No raw `fetch` in app code (sole exception: `planner-proxy.ts`).
3. **State has one writer.** Process state is owned by process singletons
   (`ServerState`, `LifecycleManager`, `TrainingExecutor`, `OutputBuffer`,
   `EventBus`, `ResourceManager`, kv/session managers — all via `get_*()` accessors,
   lock-guarded, lazy). Clients subscribe/read via endpoints; they do not write state.
4. **Versioned, not forked.** Core-stable endpoints below are the depend-on set for
   both shells. Features land on the web with a feature matrix entry; app parity is a
   tracked diff, not a second implementation.

### Core-stable endpoints (both shells depend on these)

| Group        | Endpoints                                                                                                      | Purpose                               |
| ------------ | -------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| Health       | `/health`, `/health/live`, `/health/ready`, `/health/detailed`, `/health/startup-progress`, `/health/services` | liveness/readiness, phase progress    |
| System state | `/system/metrics`, `/system/info`, `/system/disk`, `/system/lifecycle`, `/system/inference-pool`               | server state read-model               |
| Streaming    | `/system/stream` (SSE output), `/chat/stream` (SSE chat), `/errors/stream`, `/training/stream`                 | live state + generation               |
| Executor     | `/system/executor`, `/system/executor/{job_id}`, `/result`, `/purge`, `/cancel`                                | training job pool                     |
| Core flows   | `/chat`, `/inference/*`, `/models/*`, `/session/*`, `/knowledge`, `/memory`, `/settings/*`                     | feature surface shared by both shells |

### State read-model

`ServerState` is the single source of truth for runtime state (`model`, `tokenizer`,
`checkpoint`, `current_soul`, `gen_config`, `training_active`, metrics rings). It is
exposed to clients **only** through the read-model endpoints above; fields are
`AtomicRef` (`.get()`/`.set()`, one writer). Do not add a new state mutation
endpoint without a matching read-model consumer.

### SSE frame schema

Every SSE endpoint (`/chat/stream`, `/system/stream`, */stream) yields frames as
lines `data: {json}` with no `event:` name, framed per `domain/api` SSE envelope:

```json
{ "stream": "...", "phase": "token|thinking|MEMORY|TOOL|CONTROL|status|error|...", "status": "...", "data": {...} }
```

`streamSSE` reconnects on last event id; `status: "error"` frames are terminal.

### Error taxonomy

Structured errors via `classify_and_raise(e, source="router.method")`:
`E_INFRA_STARTUP` (503), `E_NOT_FOUND` (404), `E_DOMAIN` (422/400), `E_AUTH`
(401/403). The error body is **flat** (built by `schemas.common.error_response()`,
the single source of truth) — it is *not* wrapped in a `status`/`data` envelope:

```json
{ "error": "human message", "code": "E_NOT_FOUND", "details": { ... }, "correlation_id": "..." }
```

`details` and `correlation_id` are present when available (`correlation_id`
comes from `CorrelationIdMiddleware`). Clients read `j.error` / `j.code` /
`j.correlation_id` (`apps/web/lib/http-client.ts`). Shapes are pinned by
`tests/contract/test_api_envelope.py`; the executor/`/system/*` read-model
is covered by `tests/server/test_system_router.py`.

## Python SDK (`packages/sdk-py`)

| Class                         | Method count | Notes                                           |
| ----------------------------- | ------------ | ----------------------------------------------- |
| `SloughGPTClient`             | 84           | Sync client. `requests`-based.                  |
| `AsyncSloughGPTClient`        | 27           | Async mirror (`httpx`).                         |
| `Benchmark` / `SimpleTracker` | —            | Local benchmarking / metric tracking utilities. |
| `InMemoryCache`               | —            | TTL cache.                                      |

### Sync client method groups

| Group                                  | Methods                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Health & system                        | `health`, `liveness`, `readiness`, `detailed_health`, `info`, `get_system_metrics`, `get_system_info`, `get_system_disk`, `metrics`, `metrics_prometheus`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Generation                             | `generate`, `generate_stream`, `chat`, `chat_stream`, `quick_generate`, `quick_chat`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Models                                 | `list_models`, `load_model`, `unload_model`, `get_current_model`, `list_hf_models`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Sessions                               | `create_session`, `list_sessions`, `get_session`, `delete_session`, `save_session_context`, `get_session_messages`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Souls                                  | `list_souls`, `get_current_soul`, `switch_soul`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Knowledge                              | `list_knowledge`, `add_knowledge`, `delete_knowledge`, `search_knowledge`, `get_knowledge_stats`, `get_knowledge_topics`, `ingest_knowledge_url`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Tokenizer                              | `get_tokenizer_stats`, `tokenize`, `train_tokenizer`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Personality / companion                | `get_personalities`, `set_personality`, `get_companion_prompt`, `list_companion_presets`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Datasets                               | `list_datasets`, `get_dataset`, `get_dataset_stats`, `import_dataset_local`, `import_dataset_github`, `import_dataset_url`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Training                               | `start_training`, `get_training_status`, `list_training_jobs`, `delete_training_job`, `stop_training`, `pause_training`, `resume_training`, `get_training_recovery_stats`, `abandon_recovery`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Auto-train                             | `start_auto_train`, `stop_auto_train`, `get_auto_train_status`, `list_auto_train_checkpoints`, `delete_auto_train_checkpoint`, `load_auto_train_checkpoint`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Feedback & workflow                    | `record_feedback`, `get_feedback_stats`, `get_workflow_status`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Experiments                            | `create_experiment`, `list_experiments`, `get_experiment`, `log_metric`, `log_param`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Rate limit                             | `get_rate_limit_status`, `check_rate_limit`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Security                               | `get_audit_log`, `get_security_keys`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Security history (`/security/audit`)   | `history=true` reads persisted `audit.log` (survives restart); `before=<ISO timestamp>` cursor pagination; `event_type=<type>` filter; `limit` (0 = all, negative mirrors ring `[-limit:]`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Audit instrumentation (privileged ops) | `model.load`, `model.unload`, `model.quantize/dequantize/precision/download/cancel`, `soul.switch`, `soul.weights.save`, `weights.snapshot.save/load/delete`, `training.start` (detail `char`/`hf`), `training.stop`, `training.delete`, `training.checkpoint.load/delete`, `training.webhook.register/delete`, `dataset.create/update/delete/version/data.append/import/convert`, `knowledge.add/update/delete/batch.delete`, `agent.create/update/delete/execute`, `config.generation.save`, `experiment.create/delete`, `adapter.update/reset/merge/aggregate/delete/prune`, `adapter.eval.aggregate`, `tokenizer.train`, `executor.purge/cancel`, `self_train.start/stop`, `multimodal.checkpoint.load/delete`, `multimodal.reset`, `training.pause/resume` — emitted via `AuditLogger.log` (best-effort, never breaks the operation); queryable at `/security/audit?history=true&event_type=<type>` |
| Model registry                         | `list_registry_models`, `get_registry_model`, `get_registry_best`, `get_registry_stats`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Benchmark                              | `run_benchmark`, `get_benchmark_metrics`, `get_benchmark_stats`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

### Async client

Covers the same surface for the core flows: health, generation, chat, models,
souls, knowledge, metrics, workflow, feedback, training, experiments, tokenizer,
auto-train checkpoints, security keys, and the model registry.

## TypeScript SDK (`packages/sdk-ts/typescript-sdk`)

`SloughGPTClient` — 82 async methods mirroring the Python sync client, including
`generateStream`/`chatStream` (SSE), `getSecurityKeys`, and the four registry
methods (`listRegistryModels`, `getRegistryModel`, `getRegistryBest`,
`getRegistryStats`). Primary target for the React Native app (`apps/mobile`).
Also exports `useSloughGPT` (React hook) and an `index.ts` barrel.

## CLI (`sloughgpt-cli`)

```
sloughgpt-cli health | info | generate | chat | models | datasets | metrics | registry
registry actions: list | info <id> | best | stats
```

Registry commands proxy the live server registry (`GET /registry/*`); there is no
client-side registry state.

## Endpoint Coverage

The backend exposes **617 routes across 56 routers** (`apps/api/server/routers`).
The SDK covers the primary consumer-facing surface; the complete server-side route
list is documented in [`docs/routers.md`](routers.md).

| Router                                                                                                                                                                      | Covered by SDK                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Health (`/health`, `/health/detailed`)                                                                                                                                      | ✅                                                            |
| Inference (`/inference/generate`, `/inference/generate/stream`, `/chat`, `/chat/stream`)                                                                                    | ✅                                                            |
| Models (`/models`)                                                                                                                                                          | ✅                                                            |
| Souls (`/souls`)                                                                                                                                                            | ✅                                                            |
| Knowledge (`/knowledge`)                                                                                                                                                    | ✅                                                            |
| Tokenizer (`/tokenizer`)                                                                                                                                                    | ✅                                                            |
| Token tree (`/token-tree/*`)                                                                                                                                                | ✅                                                            |
| System (`/system/*`)                                                                                                                                                        | ✅                                                            |
| Datasets (`/datasets`)                                                                                                                                                      | ✅                                                            |
| Training (`/training/*`)                                                                                                                                                    | ✅                                                            |
| Feedback / workflow (`/feedback/*`, `/workflow/status`)                                                                                                                     | ✅                                                            |
| Experiments (`/experiments`)                                                                                                                                                | ✅                                                            |
| Rate limit (`/rate-limit/*`)                                                                                                                                                | ✅                                                            |
| Security (`/security/audit`, `/security/keys`)                                                                                                                              | ✅                                                            |
| Security history (`/security/audit?history=true&before=&event_type=&limit=`)                                                                                                | ✅                                                            |
| Audit instrumentation (models, souls, auto-train, datasets, kb, agents, config, experiments, adapters, lora-eval, tokenizer, system, self-train, multimodal privileged ops) | ✅                                                            |
| Registry (`/registry/models`, `/registry/models/{id}`, `/registry/best`, `/registry/stats`)                                                                                 | ✅                                                            |
| Benchmark (`/benchmark/*`)                                                                                                                                                  | ✅                                                            |
| Companion (`/companion/*`)                                                                                                                                                  | ✅                                                            |
| VM console (`/vm/run`, `/vm/builtins`, `/vm/info`, `/vm/training/jobs/{id}`)                                                                                                | Backend-only — see [`docs/VM_CONSOLE.md`](VM_CONSOLE.md)      |
| Other routers (multimodal, agents, shell, LORA eval, user adapters, system executor, etc.)                                                                                  | Backend-only — call via HTTP client or `apps/web` controllers |

### Deleted fake modules

`billing.py`, `webhooks.py`, `dashboard.py`, `auth.py`, and `registry.py` were
removed from the Python SDK. They were local in-memory implementations with **no
backend endpoints**. Any SaaS-style features live server-side only (see
`apps/api/server/routers/`). `websocket.py` remains (targets `/ws/generate`).

## References

- [`docs/routers.md`](routers.md) — backend router/endpoint reference
- `packages/sdk-py/sloughgpt_sdk/README.md` — Python SDK usage guide
- `packages/sdk-ts/typescript-sdk/README.md` — TypeScript SDK usage guide
- [`AGENTS.md`](../AGENTS.md) — architecture, conventions, and repo map

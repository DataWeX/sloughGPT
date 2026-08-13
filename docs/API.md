# SloughGPT API & SDK Documentation

Reference for the HTTP API served by `apps/api/server` and the SDKs that call it.
The SDK surface is **real** — every method maps to a live backend endpoint. There
are no simulated, in-memory, or local-only API methods.

## Response Envelope

Most routers wrap responses in a `StandardResponse` envelope:

```json
{ "status": "success", "data": { ... } }
```

List-returning endpoints put the array inside `data` (e.g. `/registry/models` →
`data: { "models": [...], "count": N }`, `/knowledge` → `data: { "items": [...], "count": N }`).

All SDK client methods unwrap this envelope automatically, so callers receive the
payload, not the envelope. Bare responses (non-enveloped lists/dicts) pass through unchanged.

## Python SDK (`packages/sdk-py`)

| Class | Method count | Notes |
|-------|-------------|-------|
| `SloughGPTClient` | 84 | Sync client. `requests`-based. |
| `AsyncSloughGPTClient` | 27 | Async mirror (`httpx`). |
| `Benchmark` / `SimpleTracker` | — | Local benchmarking / metric tracking utilities. |
| `InMemoryCache` | — | TTL cache. |

### Sync client method groups

| Group | Methods |
|-------|---------|
| Health & system | `health`, `liveness`, `readiness`, `detailed_health`, `info`, `get_system_metrics`, `get_system_info`, `get_system_disk`, `metrics`, `metrics_prometheus` |
| Generation | `generate`, `generate_stream`, `chat`, `chat_stream`, `quick_generate`, `quick_chat` |
| Models | `list_models`, `load_model`, `unload_model`, `get_current_model`, `list_hf_models` |
| Sessions | `create_session`, `list_sessions`, `get_session`, `delete_session`, `save_session_context`, `get_session_messages` |
| Souls | `list_souls`, `get_current_soul`, `switch_soul` |
| Knowledge | `list_knowledge`, `add_knowledge`, `delete_knowledge`, `search_knowledge`, `get_knowledge_stats`, `get_knowledge_topics`, `ingest_knowledge_url` |
| Tokenizer | `get_tokenizer_stats`, `tokenize`, `train_tokenizer` |
| Personality / companion | `get_personalities`, `set_personality`, `get_companion_prompt`, `list_companion_presets` |
| Datasets | `list_datasets`, `get_dataset`, `get_dataset_stats`, `import_dataset_local`, `import_dataset_github`, `import_dataset_url` |
| Training | `start_training`, `get_training_status`, `list_training_jobs`, `delete_training_job`, `stop_training`, `pause_training`, `resume_training`, `get_training_recovery_stats`, `abandon_recovery` |
| Auto-train | `start_auto_train`, `stop_auto_train`, `get_auto_train_status`, `list_auto_train_checkpoints`, `delete_auto_train_checkpoint`, `load_auto_train_checkpoint` |
| Feedback & workflow | `record_feedback`, `get_feedback_stats`, `get_workflow_status` |
| Experiments | `create_experiment`, `list_experiments`, `get_experiment`, `log_metric`, `log_param` |
| Rate limit | `get_rate_limit_status`, `check_rate_limit` |
| Security | `get_audit_log`, `get_security_keys` |
| Security history (`/security/audit`) | `history=true` reads persisted `audit.log` (survives restart); `before=<ISO timestamp>` cursor pagination; `event_type=<type>` filter; `limit` (0 = all, negative mirrors ring `[-limit:]`) |
| Audit instrumentation (privileged ops) | `model.load`, `model.unload`, `model.quantize/dequantize/precision/download/cancel`, `soul.switch`, `soul.weights.save`, `weights.snapshot.save/load/delete`, `training.start` (detail `char`/`hf`), `training.stop`, `training.delete`, `training.checkpoint.load/delete`, `training.webhook.register/delete`, `dataset.create/update/delete/version/data.append/import/convert`, `knowledge.add/update/delete/batch.delete`, `agent.create/update/delete/execute`, `config.generation.save`, `experiment.create/delete`, `adapter.update/reset/merge/aggregate/delete/prune`, `adapter.eval.aggregate`, `tokenizer.train`, `executor.purge/cancel`, `self_train.start/stop`, `multimodal.checkpoint.load/delete`, `multimodal.reset`, `training.pause/resume` — emitted via `AuditLogger.log` (best-effort, never breaks the operation); queryable at `/security/audit?history=true&event_type=<type>` |
| Model registry | `list_registry_models`, `get_registry_model`, `get_registry_best`, `get_registry_stats` |
| Benchmark | `run_benchmark`, `get_benchmark_metrics`, `get_benchmark_stats` |

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

The backend exposes **344 routes across 37 routers** (`apps/api/server/routers`).
The SDK covers the primary consumer-facing surface; the complete server-side route
list is documented in [`docs/routers.md`](routers.md).

| Router | Covered by SDK |
|--------|----------------|
| Health (`/health`, `/health/detailed`) | ✅ |
| Inference (`/inference/generate`, `/inference/generate/stream`, `/chat`, `/chat/stream`) | ✅ |
| Models (`/models`) | ✅ |
| Souls (`/souls`) | ✅ |
| Knowledge (`/knowledge`) | ✅ |
| Tokenizer (`/tokenizer`) | ✅ |
| Token tree (`/token-tree/*`) | ✅ |
| System (`/system/*`) | ✅ |
| Datasets (`/datasets`) | ✅ |
| Training (`/training/*`) + auto-train (`/auto-train/*`) | ✅ |
| Feedback / workflow (`/feedback/*`, `/workflow/status`) | ✅ |
| Experiments (`/experiments`) | ✅ |
| Rate limit (`/rate-limit/*`) | ✅ |
| Security (`/security/audit`, `/security/keys`) | ✅ |
| Security history (`/security/audit?history=true&before=&event_type=&limit=`) | ✅ |
| Audit instrumentation (models, souls, auto-train, datasets, kb, agents, config, experiments, adapters, lora-eval, tokenizer, system, self-train, multimodal privileged ops) | ✅ |
| Registry (`/registry/models`, `/registry/models/{id}`, `/registry/best`, `/registry/stats`) | ✅ |
| Benchmark (`/benchmark/*`) | ✅ |
| Companion (`/companion/*`) | ✅ |
| VM console (`/vm/run`, `/vm/builtins`, `/vm/info`, `/vm/training/jobs/{id}`) | Backend-only — see [`docs/VM_CONSOLE.md`](VM_CONSOLE.md) |
| Other routers (multimodal, agents, shell, LORA eval, user adapters, system executor, etc.) | Backend-only — call via HTTP client or `apps/web` controllers |

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

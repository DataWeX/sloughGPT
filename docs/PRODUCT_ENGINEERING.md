# Product Engineering Map

Source of truth for what we build, why, and how it maps to code.
Reference this BEFORE writing any new feature, router, or page.

**Primary persona:** Alex the Hobbyist (see `USER_PERSONA.md`)
**User flows:** See `UX_FLOWS.md`

---

## The model (model, not net)

sloughGPT is a general-AI **model**, not a pile of layers. SloNet (pure-NumPy
Tensor/autograd, SloTransformer, TokenTree, `.soul`/`.slnc`) is machinery in
service of the model — never the definition of it. The model is defined by:

- **Capabilities** — what it can do (converse, remember, use tools, learn).
- **I/O contract** — declared inputs/outputs, versioned with the weights.
- **Evals** — reproducible checks that prove each capability per release.
- **Versioning** — weights + tokenizer + contract ship as one unit.

Borrowed weights (HF conversion, distillation) are bootstrap, not destination.
New work must move at least one of owned architecture, owned objective
(beyond next-token prediction), owned data toward the model — or say why not.

---

## CCGT fairness rule (Cognitive, Core, Gateway, Training)

The only four domains. Every piece ships all four or it doesn't ship:

| Domain        | Question it answers                                                                                 |
| ------------- | --------------------------------------------------------------------------------------------------- |
| **Cognitive** | How does this integrate with reasoning, memory, souls, attention?                                   |
| **Core**      | Where does it live in the serving core (engine, providers, KV, queues)?                             |
| **Gateway**   | How is this exposed and optimized at the edge (filtering, streaming relay, timeouts, load shaping)? |
| **Training**  | How is this learned or improved (objective, data, loop)?                                            |

No Core-only work that leaves the other three behind. The gateway is a
domain, not plumbing — it is how we optimize the core infra stack's exposure,
so it must be built well: generic byte-relay, zero-buffer streaming, strict
filters, never duplicated model logic.

---

## Engine-first

`InferenceEngine` + `SloNetServer` is the core everything flows through.
Harden it before adding features. Perf budgets are acceptance gates:

- Facade dispatch overhead: sub-millisecond (respond/stream).
- KV-cache session reuse across turns; prefix-stable prompts where possible.
- Zero-buffer token streaming end to end; per-route timeouts.
- Benchmark before/after (`scripts/benchmark_latency.py`,
  `scripts/benchmark_kv_cache.py`); >20% regression reverts.

---

## Client / core separation

Clients (web, CLI, mobile, SDK) stay out of core infra so both sides stay
writable — and the Gateway domain, which owns edge optimization, talks to core
only through the same stable interfaces:

- Web → FastAPI over HTTP, exclusively via `http-client.ts`
  (`apiGet/apiPost/...`) — never raw `fetch`, never `domain/*` imports.
- API server → engine over the length-prefixed TCP protocol
  (`inference_protocol.py`) via `InferenceClient`, or in-process through the
  provider registry — never by reaching into engine internals.
- Gateway → core as a generic byte-relay with filters (no model logic
  duplicated at the edge); same interface discipline as any client.
- No client imports from `domain.*_internal*`; no core file imports from
  `apps/web`, `apps/cli`, or `apps/gateway`.

---

## Core Features (what Alex actually uses)

| #   | Feature       | What Alex does                         | Domain module                 | API prefix                       | Frontend page |
| --- | ------------- | -------------------------------------- | ----------------------------- | -------------------------------- | ------------- |
| 1   | **Chat**      | Talks to AI, AI remembers              | `chat`, `memory`, `companion` | `/chat`, `/memory`, `/companion` | `/chat`       |
| 2   | **Train**     | Picks data, clicks train, sees results | `training`                    | `/training`                      | `/training`   |
| 3   | **Datasets**  | Imports data to train on               | `training` (DatasetManager)   | `/datasets`                      | `/datasets`   |
| 4   | **Models**    | Switches between trained models        | `models`, `inference`         | `/models`, `/infer`              | `/models`     |
| 5   | **Souls**     | Gives AI a personality                 | `soul`, `ai_personality`      | `/souls`                         | `/souls`      |
| 6   | **Knowledge** | AI learns from files/docs              | `knowledge`                   | `/knowledge`                     | `/knowledge`  |
| 7   | **Tools**     | Writing, translate, rewrite, etc.      | `tools`                       | `/tools`                         | `/tools`      |

## Power User Features (Alex might discover later)

| #   | Feature       | What Alex does                | Domain module | API prefix   | Frontend page |
| --- | ------------- | ----------------------------- | ------------- | ------------ | ------------- |
| 8   | **Benchmark** | Checks if model improved      | `benchmark`   | `/benchmark` | `/benchmark`  |
| 9   | **Feedback**  | Rates AI responses, AI learns | `feedback`    | `/feedback`  | `/feedback`   |
| 10  | **Settings**  | Tweak behavior                | `settings`    | `/settings`  | `/settings`   |

## System Features (Alex never sees)

| #   | Feature        | What it does    | Domain module    | API prefix | Frontend page |
| --- | -------------- | --------------- | ---------------- | ---------- | ------------- |
| 11  | **Shell**      | Terminal access | `shell`          | `/shell`   | `/shell`      |
| 12  | **VM**         | Linux VM        | `shell`          | `/vm`      | `/vm`         |
| 13  | **Monitoring** | System health   | `infrastructure` | `/system`  | `/monitoring` |
| 14  | **Developer**  | API调试         | `infrastructure` | `/infer`   | `/developer`  |

## Experimental / Not Yet Integrated

| #   | Feature              | Status           | Domain module          | Notes                                                              |
| --- | -------------------- | ---------------- | ---------------------- | ------------------------------------------------------------------ |
| 15  | **Consciousness**    | Wired           | `consciousness`        | Level-gated post-gen wiring (SloEngine hook + router `_run_post_gen_tasks`); narrative via reasoning_chain + `CONSCIOUSNESS` SSE; 27 sub-pages still need flow review |
| 16  | **Voice**            | Partial          | `voice`                | TTS works, phoneme encode works, but router bypasses domain        |
| 17  | **Phoneme Learning** | Over-built       | `voice` (phoneme)      | 47 components, 12 tabs — needs flow restructure                    |
| 18  | **Tokenizer**        | Over-built       | `training` (tokenizer) | 9 tabs, 12 sub-cards — needs consolidation                         |
| 19  | **Companion**        | Built            | `companion`            | AI companion with personality                                      |
| 20  | **Agents**           | Built            | `agents`               | Agentic tool execution                                             |

---

## Feature-Level API Rule

Each feature must have a **single engine/service class** in its domain module that
wraps all capabilities. The router delegates to this engine. The frontend calls
feature-level endpoints, not implementation-level endpoints.

### Good examples (reference these)

**Tools** — `domain/tools/`:

```
ToolsEngine
  ├── get_tool_profiles() → list of available tools
  ├── render_prompt(tool_id, params) → rendered prompt
  └── generate(tool_id, params, callbacks) → streamed output

Router: GET /tools, POST /tools/{id}/generate
Frontend: tools-controller.ts → 1 page with tool selector
```

**Collections** — `domain/collections/`:

```
Registry
  ├── list_pipelines()
  ├── create_pipeline(config)
  ├── run_pipeline(id)
  └── get_stats()

Router: GET /collections, POST /collections/create, POST /collections/run
Frontend: collections page (single page)
```

### Bad examples (fix these)

> **Status 2026-09-29:** Voice/tokenizer/token_tree/kb bypasses are fixed (facade
> or engine imports). `training/router.py` now delegates everything to
> `TrainingEngine` (0 `_internal`, see Router Playbook below). Remaining: the
> engines' own internals — `self_train`/`lora_eval` counts are inside
> concurrent WIP files.

**Voice** — 3 routers, domain bypassed *(historical example — fixed 2026-09-29 per the status note above; the routers now import `domain.voice` / `domain.training` facades)*:

```
voice.py router → imports from domain.multimodal._internal.tts (WRONG)
tokenizer.py router → imports from domain.training._internal.tokenizer_manager (WRONG)
token_tree.py router → imports from domain.training._internal.token_tree_manager (WRONG)

Should be:
VoiceEngine → wraps TTS + phoneme + recognition from domain.voice
TokenizerEngine → wraps tokenizer from domain.training
Single router per engine, not per implementation module
```

**Knowledge** — 1 monolith router, domain bypassed *(historical example — fixed 2026-09-29 per the status note above)*:

```
kb.py (1399 lines) → imports from domain.learner._internal.knowledge (WRONG)
7 features inline: CRUD, RAG, training, spaced repetition, files, categorization, adapter

Should be:
KnowledgeEngine → wraps KnowledgeMemory + KnowledgeIngestor + DataFilter from domain.knowledge
Router delegates to engine methods
```

**Training** — 7 routers, no unifying engine:

```
training/router.py, self_train.py, lora_eval.py, cloud_training.py,
tokenizer.py, token_tree.py → all import from domain.training._internal.* (WRONG)

Should be:
TrainingEngine → wraps DatasetManager + TrainingPipeline + ModelManager from domain.training
Single /training router with sub-paths for each feature
```

---

## Router Playbook

How to build a Python API router in this repo. FastAPI's `@router.get` /
`@router.post` decorators are the right idiom — the rules are about what lives
in the handler, not the decorator. Endpoint inventory: [`routers.md`](routers.md).
Before adding a capability at all, the contract-first pattern in
[`TRANSPORT_PROJECTIONS.md`](TRANSPORT_PROJECTIONS.md) applies (descriptor →
projected route, not a new endpoint project). Philosophy:
[`PYTHON_FIRST.md`](PYTHON_FIRST.md) §4 "Web router is just plumbing".

**1. One package per feature.** `apps/api/server/<feature>/{router.py, schemas.py, ...}`
— reference: `apps/api/server/training/`.

**2. Handler = HTTP only.** Parse/validate → call the engine → envelope.
Zero business logic, zero `_internal` imports (`grep -c _internal` must be 0):

```python
from schemas.common import raise_error, safe_audit_log, success_response
from domain.training import get_training_engine   # facade; function-local, lazy


@router.delete("/training/checkpoints/{name}")
async def training_delete_checkpoint(name: str):
    engine = get_training_engine()
    if not engine.is_valid_checkpoint_name(name):
        raise_error("Invalid checkpoint name", "E_BAD_REQUEST", status_code=400)
    deleted = await engine.delete_checkpoint(name)
    if deleted:
        safe_audit_log("training.checkpoint.delete", resource=name, detail="deleted")
    return success_response(data={"deleted": deleted, "name": name})
```

**3. One engine per feature** (rule above). Engines may import `_internal` —
they own the domain. Routers never do.

**4. Error classification.**
- Request problem (bad input, missing resource) → `raise_error(msg, "E_*", status_code=...)`
  at the edge; no exception machinery.
- Unexpected exception → let it bubble to the endpoint, then
  `classify_and_raise(e, source="feature.op")`.
- Expected domain failure → engine returns an envelope with
  `metadata.code = "E_*"`; router passes it through.

**5. Envelopes + audit.** Always `success_response(data=...)` from
`schemas.common`; durable side effects get `safe_audit_log(...)`.

**6. Async.** `async def` endpoints; blocking work runs via the training
executor or an engine-managed thread — never block the event loop.

**7. Registration.** Add to `get_all_routers()` in
`apps/api/server/routers/__init__.py` (health/status register pre-lifespan).

**8. Tests patch the facade.** Patch `domain.<feature>.get_x` / the engine
method — the symbol the router actually looks up — never router-module
symbols: renaming a router import breaks `mock.patch("training.router.x")`
fixtures with `AttributeError` at setup (see `test_training_recovery_router.py`).

**Checklist:** `grep -c _internal <router>` = 0 → registered → ruff → router's
tests green.

---

## UI Design Rules (from USER_PERSONA.md)

1. **One-click training** — Default settings work for 90% of cases
2. **Jargon-free** — Never say "LoRA", "GRPO", "KL coefficient" in UI
3. **Progressive disclosure** — Results first, details on click, expert settings behind toggle
4. **Plain language** — "Training complete! Your AI now knows Shakespeare" not "loss=1.2345"
5. **Visual, not numerical** — Loss chart with down arrow = good
6. **Features are flows, not tools** — Don't expose 12 phoneme tools, expose a pronunciation learning flow
7. **One app-wide banner** — journey-blocking alerts (training failed, backend down) go through the global banner system (`useBannerStore` + `<GlobalBanner />` in `AppLayout`) so they survive navigation; toasts only for transient confirmations

---

## Route Map (what exists vs what should exist)

### Current: 85+ frontend pages, 58 routers

### Target: ~20 frontend pages, ~15 routers

**Status legend:** ✅ done / 🔧 partial (nav-level) / ⏳ pending

| Target Page   | Current Pages That Merge                                                                   | Routers That Merge                                                            | Status                                                                                  |
| ------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `/chat`       | `/chat` (already clean)                                                                    | `inference.py`                                                                | ✅                                                                                      |
| `/training`   | `/training` + `/self-train` + `/lora-eval` + `/auto-train` + `/datasets` + 12 sub-pages    | `training/router.py` + `self_train.py` + `lora_eval.py` + `cloud_training.py` | 🔧 sidebar keeps `/training` + `/training/runs`; `/auto-train` `/self-train` headless   |
| `/datasets`   | `/datasets` + `/dataset/[id]`                                                              | `datasets.py`                                                                 | ✅                                                                                      |
| `/models`     | `/models` + `/model/[id]` + `/infer` + `/registry`                                         | `models.py` + `infer.py` + `registry.py`                                      | ⏳                                                                                      |
| `/souls`      | `/souls` + `/personality` + `/companion`                                                   | `souls.py` + `companion.py`                                                   | ⏳                                                                                      |
| `/knowledge`  | `/knowledge` + `/kb` + `/docstore` + `/learn`                                              | `kb.py` + `learner.py`                                                        | ⏳                                                                                      |
| `/tools`      | `/tools` + `/writing` + `/rewrite` + `/translate` + `/explain` + `/brainstorm` + `/decide` | `tools.py` (already clean)                                                    | ✅ all 8 tool pages removed → chat `ModeBar` (`/chat?mode=<mode>`), API-only `tools.py` |
| `/benchmark`  | `/benchmark` + `/experiments`                                                              | `benchmark.py` + `experiments.py`                                             | 🔧 headless (removed from sidebar — API-only `POST /benchmark/score`, no UI for Alex)   |
| `/feedback`   | `/feedback` + `/meta-weights` + `/workflow`                                                | `feedback.py` + `meta_weights.py` + `workflow.py`                             | 🔧 kept in sidebar (Venn center for Maya/Jon)                                           |
| `/settings`   | `/settings` + `/admin` + `/security` + `/api-keys` + `/rate-limit`                         | `settings.py` + `security.py`                                                 | ⏳                                                                                      |
| `/shell`      | `/shell` + `/vm` + `/developer`                                                            | `shell.py` + `vm.py`                                                          | ⏳                                                                                      |
| `/voice`      | `/voice` + `/phoneme` + `/tokenizer` + `/token-tree`                                       | `voice.py` + `tokenizer.py` + `token_tree.py`                                 | 🔧 `/voice` page removed → `/chat?mode=talk`; phoneme left alone per directive          |
| `/memory`     | `/memory` (already clean)                                                                  | `memory.py`                                                                   | ✅                                                                                      |
| `/files`      | `/files` + `/images`                                                                       | `files.py` + `images.py`                                                      | ⏳                                                                                      |
| `/monitoring` | `/monitoring` + `/errors` + `/session`                                                     | `system.py` + `errors.py` + `session.py`                                      | ⏳                                                                                      |

> **Persona rule for benchmarking (from USER_PERSONA.md):** `/benchmark` is hidden from Alex (he gets an inline training verdict) and there is no dedicated benchmark UI — `POST /benchmark/score` + `config/bench_weights.yaml` is the headless API for Maya/Jon CI gating. Do not reintroduce a benchmark page into the sidebar.

---

## Dead Code to Remove

| What                            | Why                                       | Action                                      |
| ------------------------------- | ----------------------------------------- | ------------------------------------------- |
| `routers/metrics.py`            | No frontend consumer, infrastructure-only | Delete or move to internal                  |
| `routers/collections.py`        | No frontend page or controller            | Delete (feature unused)                     |
| `routers/feeds.py`              | RSS feeds, no web frontend consumer       | Move to internal or delete                  |
| `routers/api_keys.py`           | Duplicate of `security.py` (same prefix)  | Merge into security.py, delete api_keys.py  |
| `routers/session_store.py`      | Utility module, not a router              | Move to `controllers/` or `infrastructure/` |
| Phoneme: 47 components          | Over-built for Alex's needs               | Keep components, restructure page into flow |
| Tokenizer: 9 tabs, 12 sub-cards | Over-built                                | Consolidate into 2-3 sections               |
| Consciousness: 27 sub-pages     | Engine wired; pages still over-built        | Keep pages, review flows against UX_FLOWS.md |

---

## Build Order (what to do next)

1. **Write this doc** ✅ (you're reading it)
2. **Quick cleanup** ✅ — dead routers removed (metrics, collections, feeds, session_store); nav consolidated: `/benchmark` headless, training sub-pages integrated, 8 tool pages removed → chat `ModeBar` (`/chat?mode=<mode>`), `/tools` grid removed
3. **Feature engines** ✅ — VoiceEngine, KnowledgeEngine, TrainingEngine (ToolsEngine pattern) + `get_*_engine` facade exports, routers wired
4. **Wire routers** ✅ — routers import public `domain.<pkg>` facades only (0 `_internal` imports across all 58 router files, ratcheted by `tests/contract/test_router_boundary.py` with an empty baseline; facade lazy exports added for heavy symbols; tests patch facade paths)
5. **UI flows** — restructure phoneme, knowledge, training pages to follow UX_FLOWS.md
6. **Consciousness** ✅ — wire cognitive engine to inference pipeline (SloEngine post-gen hook + router single process, level-gated via `ConsciousnessConfig.level`; narrative surfaced through `reasoning_chain` and `CONSCIOUSNESS complete` SSE)
7. **Test** — verify all user journeys pass

---

_This doc is the source of truth. When in doubt, reference it. When building, follow it._

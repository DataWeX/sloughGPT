# Product Engineering Map

Source of truth for what we build, why, and how it maps to code.
Reference this BEFORE writing any new feature, router, or page.

**Primary persona:** Alex the Hobbyist (see `USER_PERSONA.md`)
**User flows:** See `UX_FLOWS.md`

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
| 15  | **Consciousness**    | Built, not wired | `consciousness`        | 27 sub-pages exist but cognitive engine not connected to inference |
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

**Voice** — 3 routers, domain bypassed:

```
voice.py router → imports from domain.multimodal._internal.tts (WRONG)
tokenizer.py router → imports from domain.training._internal.tokenizer_manager (WRONG)
token_tree.py router → imports from domain.training._internal.token_tree_manager (WRONG)

Should be:
VoiceEngine → wraps TTS + phoneme + recognition from domain.voice
TokenizerEngine → wraps tokenizer from domain.training
Single router per engine, not per implementation module
```

**Knowledge** — 1 monolith router, domain bypassed:

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

| Target Page   | Current Pages That Merge                                                                   | Routers That Merge                                                            |
| ------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `/chat`       | `/chat` (already clean)                                                                    | `inference.py`                                                                |
| `/training`   | `/training` + `/self-train` + `/lora-eval` + `/auto-train` + `/datasets` + 12 sub-pages    | `training/router.py` + `self_train.py` + `lora_eval.py` + `cloud_training.py` |
| `/datasets`   | `/datasets` + `/dataset/[id]`                                                              | `datasets.py`                                                                 |
| `/models`     | `/models` + `/model/[id]` + `/infer` + `/registry`                                         | `models.py` + `infer.py` + `registry.py`                                      |
| `/souls`      | `/souls` + `/personality` + `/companion`                                                   | `souls.py` + `companion.py`                                                   |
| `/knowledge`  | `/knowledge` + `/kb` + `/docstore` + `/learn`                                              | `kb.py` + `learner.py`                                                        |
| `/tools`      | `/tools` + `/writing` + `/rewrite` + `/translate` + `/explain` + `/brainstorm` + `/decide` | `tools.py` (already clean)                                                    |
| `/benchmark`  | `/benchmark` + `/experiments`                                                              | `benchmark.py` + `experiments.py`                                             |
| `/feedback`   | `/feedback` + `/meta-weights` + `/workflow`                                                | `feedback.py` + `meta_weights.py` + `workflow.py`                             |
| `/settings`   | `/settings` + `/admin` + `/security` + `/api-keys` + `/rate-limit`                         | `settings.py` + `security.py`                                                 |
| `/shell`      | `/shell` + `/vm` + `/developer`                                                            | `shell.py` + `vm.py`                                                          |
| `/voice`      | `/voice` + `/phoneme` + `/tokenizer` + `/token-tree`                                       | `voice.py` + `tokenizer.py` + `token_tree.py`                                 |
| `/memory`     | `/memory` (already clean)                                                                  | `memory.py`                                                                   |
| `/files`      | `/files` + `/images`                                                                       | `files.py` + `images.py`                                                      |
| `/monitoring` | `/monitoring` + `/errors` + `/session`                                                     | `system.py` + `errors.py` + `session.py`                                      |

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
| Consciousness: 27 sub-pages     | Premature — engine not wired              | Keep pages, wire engine first               |

---

## Build Order (what to do next)

1. **Write this doc** ✅ (you're reading it)
2. **Quick cleanup** — delete dead routers, merge duplicates
3. **Feature engines** — VoiceEngine, KnowledgeEngine, TrainingEngine (follow ToolsEngine pattern)
4. **Wire routers** — point routers at engines, not internals
5. **UI flows** — restructure phoneme, knowledge, training pages to follow UX_FLOWS.md
6. **Consciousness** — wire cognitive engine to inference pipeline
7. **Test** — verify all user journeys pass

---

_This doc is the source of truth. When in doubt, reference it. When building, follow it._

# Agents

## Session Checklist
Every session MUST do these three things:
1. `notes new "Title" --tags area,subarea --status wip` — first action
2. Work happens
3. `notes edit <id> --status done --body "Summary"` — before done

## Doc-First Workflow
Before any edit, read relevant docs:
- Frontend → `docs/UI_INTEGRATION_README.md`, `docs/API.md`
- Backend → `docs/routers.md`, `docs/API.md`
- Core → `docs/DEVELOPER_GUIDE.md`
- Infra → `docs/DEPLOYMENT.md`
- CLI → `docs/integration/CLI_README.md`
- Config → `docs/ENVIRONMENT.md`

## Development Principles

### Style
- No casual language — avoid contractions, slang, emojis, exclamation marks
- No unnecessary output — don't explain what you're about to do unless asked
- No confirmation dialogs — just do the work, show results
- Tables over prose — when listing items, use tables not paragraphs
- Code over explanation — show the diff, not a description of the diff

### Engineering Standards
- **Tested** — Don't assume it works, verify it works
- **Documented** — Code without docs is technical debt
- **Reversible** — Can roll back if broken
- **Complete** — Edge cases matter

### No Corner Cutting
- "It works on my machine" is not verification
- "We can fix it later" delays inevitable debt
- Write tests first or immediately after
- Document every public function
- Use config over magic values
- Handle errors explicitly

### No Mocks — Everything Must Be Programmatic
Every feature must be real, computed, and programmable. No hardcoded lookup tables, no text-file mocks.

**Test**: "If I add a new model / new input / new edge case, does it work without code changes?" If no, it's a mock.

### No Breaking UI Changes
- Use targeted edits instead of rewriting entire components
- Don't change existing user-facing behavior without being asked
- Don't move or remove existing UI elements
- When asked to "continue building" a feature, add new capability without altering existing UX
- Build up, not overhaul — add new cards/sections below existing content

### Infrastructure Before Endpoints
1. Core module / function (testable standalone)
2. CLI wrapper (human-friendly interface)
3. API endpoint (thin HTTP wrapper calling the module)

### UX First — No API Complexity for Users
- Click-and-done — User clicks a button, it works
- Auto-magic — RAG, memory, context enable themselves on first use
- Core logic handles complexity — APIs are thin wrappers only

## Repo Structure

```
sloughGPT/
├── apps/
│   ├── api/server/              # FastAPI backend
│   │   ├── main.py              # Entry point, all routes registered
│   │   └── routers/             # Route modules (41 routers, 302 handlers)
│   ├── web/                     # Next.js frontend (app router)
│   │   ├── app/(app)/           # Authenticated pages
│   │   ├── components/          # Reusable UI components
│   │   ├── features/chat/       # Chat feature-folder
│   │   ├── lib/                 # Utilities, controllers, stores
│   │   ├── hooks/               # React hooks
│   │   └── contexts/            # React contexts
│   └── mobile/                  # React Native (Expo)
└── packages/
    └── core-py/                 # Python core logic
        └── domains/             # Business logic modules
            ├── infrastructure/  # ProcessGuard, ModelServer, TaskQueue, CancelManager
            ├── inference/       # Vector store, KV cache, providers
            ├── training/        # Training pipelines, executor
            ├── feedback/        # LoRA eval, per-user adapter
            └── core/            # Soul engine, inference
```

## Commands

```bash
# Start everything (API + Web)
make stack

# Start API server only
make api

# Start frontend dev only
make web

# Type check
make tsc

# Python test (parallel)
make test-py ARGS="tests/test_file.py -x -q"

# Python syntax check
python3 -m py_compile <file>

# Web tests (targeted)
npm run test:lib      # pure logic (fastest)
npm run test:components  # UI components
npm run test:changed  # only changed files

# Full web test suite
npm run test          # ~3 min — pre-push only

# Clear Python bytecode cache
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null
```

## Key Files

### Backend
| File | Purpose |
|------|---------|
| `apps/api/server/main.py` | FastAPI entry; registers routers; `/session/{id}/context`, `/session/{id}/regenerate`, `/feedback/workflow-record` |
| `apps/api/server/routers/auto_train.py` | Unified teacher-student training; SSE with phases; GPT2 teacher + LSTM student |
| `apps/api/server/routers/souls.py` | Soul CRUD and switching; `traits` field; `checkpoint_name` param |
| `apps/api/server/routers/lora_eval.py` | LoRA evaluation endpoints |
| `apps/api/server/routers/user_adapters.py` | Per-user adapter CRUD; aggregation with eval |
| `apps/api/server/training/router.py` | Training jobs; `POST /training/start`, `GET /training/jobs`; CancelManager wired |
| `apps/api/server/routers/inference.py` | Chat, generate, regenerate; CancelManager on streaming |
| `apps/api/server/controllers/health.py` | Health endpoint; `_get_model_info()` checks ModelRegistry |
| `apps/api/server/infrastructure/startup.py` | `StartupOrchestrator`; lazy-guard autoload; background preload |

### Core Python
| File | Purpose |
|------|---------|
| `domains/infrastructure/cancel_manager.py` | CancelManager: register/start/finish/cancel ops |
| `domains/infrastructure/process_guard.py` | ProcessGuard: subprocess crash isolation, auto-restart |
| `domains/infrastructure/model_server.py` | ModelServer: semaphore, circuit breaker, idle unload, guard/local backends |
| `domains/infrastructure/model_registry.py` | ModelRegistry: composable model registry with TTL cache |
| `domains/infrastructure/task_queue.py` | InProcessTaskQueue: async priority queue with dependencies |
| `domains/infrastructure/server_state.py` | AtomicRef, ServerState: thread-safe server state |
| `domains/infrastructure/download_manager.py` | DownloadManager: resumable downloads with CancelManager |
| `domains/inference/slonet_provider.py` | SloNetChatProvider: model loading, generation, KV cache |
| `domains/training/train_pipeline.py` | SloughGPTTrainer: char-level training |
| `domains/training/sequence.py` | TrainingSequence: GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE |
| `domains/training/executor.py` | TrainingExecutor: background job execution |

### Frontend
| File | Purpose |
|------|---------|
| `apps/web/lib/api.ts` | Frontend API client (all backend calls) |
| `apps/web/lib/feedback-store.ts` | Zustand store for feedback state |
| `apps/web/lib/operations-store.ts` | Zustand vanilla store for operations tracking |
| `apps/web/hooks/useOperations.ts` | React hook with auto-polling lifecycle |
| `apps/web/app/(app)/chat/page.tsx` | Main chat page; `messagesRef` tracks live content |
| `apps/web/app/(app)/auto-train/page.tsx` | Auto-train UI; soul selector; loss curve; checkpoint catalog |
| `apps/web/app/(app)/models/page.tsx` | Model catalog; soul switcher with checkpoints submenu |
| `apps/web/app/(app)/training/page.tsx` | Training page; method selector; loss chart; checkpoint management |
| `apps/web/components/chat/` | Chat components (ChatInput, MessageBubble, ChatToolbar, etc.) |

## Infrastructure

### Process Management Stack
```
StartupOrchestrator (startup.py)
 ├── ModelRegistry (model_registry.py)
 │    └── ModelServer (model_server.py)
 │         ├── CircuitBreaker: 3 failures → 30s open → half-open
 │         ├── GuardBackend → ProcessGuard (process_guard.py)
 │         │                      └── ModelWorkerProcess (model_worker.py)
 │         ├── LocalBackend → model.generate()
 │         ├── IdleManager: unload after 1800s idle
 │         └── PriorityRequestQueue
 ├── InferencePool (inference_pool.py) — ThreadPoolExecutor
 ├── TrainingExecutor (executor.py) — ThreadPoolExecutor
 └── ServerState (server_state.py) — AtomicRef, metrics
```

### CancelManager API
```python
mgr = get_cancel_manager()
op_id = mgr.register(OpType.TRAINING, "train job 1")
mgr.start(op_id)
# ... work ...
mgr.finish(op_id)  # empty error = COMPLETED
mgr.finish(op_id, error="timeout")  # non-empty = FAILED
mgr.cancel(op_id)
mgr.list_active()
mgr.purge()
```

### Wiring Rules
- CancelManager on all long-running ops (training, inference, downloads, batch import)
- `_finish_job()` helper in training/router.py sets status + calls mgr.finish()
- Guard stopped before parent preload to avoid double-memory OOM
- ProcessGuard `_monitor_loop` snapshots `_worker` under restart_lock

## Testing Strategy

### Speed Tiers
| Command | Time | Use when |
|---------|------|----------|
| `npx tsc --noEmit` | 5-10s | Every edit |
| `npm run test:lib` | 10-20s | Changing lib/ controllers/utils |
| `npm run test:components` | 40-60s | Changing components/ |
| `npm run test:changed` | 20-40s | Quick pre-commit check |
| `npm run test` | 150-200s | Pre-push / CI only |
| `make test-py` | 2-3 min | Python full suite |

### Python Test Files
| File | Tests | What |
|------|-------|------|
| `test_cancel_manager.py` | 28 | CancelManager CRUD, OpType, OpStatus |
| `test_train_pipeline.py` | 154 | Training pipeline, personality e2e |
| `test_server_integration.py` | 57 | ModelRegistry, ModelServer, CircuitBreaker |
| `test_shell_runtime.py` | 34 | Shell start phases, probe, timeout |
| `test_shell_repl.py` | 148 | Shell commands, pipelines, env, pager |

### Frontend Test Files
| Pattern | What |
|---------|------|
| `lib/*.test.ts` | Controllers, stores, utilities |
| `components/*.test.tsx` | UI components |
| `hooks/*.test.ts` | React hooks |
| `app/**/*.test.tsx` | Page components |

### Before Submitting
1. Clear pycache: `find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null`
2. Syntax check: `python3 -m py_compile <file>` / `npx tsc --noEmit`
3. Run targeted tests first, full suite last

## UI Design System — Noir Violet

### Colors (RGB triples)
| Role | Light | Dark |
|------|-------|------|
| Primary | `124 82 196` (violet) | `192 170 244` (lilac) |
| Accent | `236 145 95` (terracotta) | `240 176 130` (peach) |
| Success | `52 176 125` | `72 192 140` |
| Warning | `236 168 60` | `240 192 80` |
| Destructive | `220 80 90` | `235 100 110` |

### Type Scale
| Role | Class | Usage |
|------|-------|-------|
| Page title | `sl-h1` / `AppRouteHeaderLead` | One per page |
| Section title | `text-base font-medium` | Card headers |
| Body | `text-sm` | Primary content |
| Caption | `text-xs text-muted-foreground` | Timestamps, secondary |
| Badge | `text-[10px] font-medium` | Status badges, tags |

### Page Template
```tsx
<div className="sl-page mx-auto max-w-4xl">
  <AppRouteHeader
    left={<AppRouteHeaderLead title="..." subtitle="..." />}
  />
  <div className="space-y-4">
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Section</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm">Content</p>
      </CardContent>
    </Card>
  </div>
</div>
```

### Rules
- `sl-page mx-auto max-w-4xl` wrapper
- `space-y-4` between sections
- `text-sm` for body, `text-xs text-muted-foreground` for meta
- Never use `text-lg` or `text-2xl` in body
- Never use hardcoded colors — use CSS variables
- Never use arbitrary spacing — use tokens

## Critical Context

- CancelManager: `domains/infrastructure/cancel_manager.py` — singleton, OpType, OpStatus
- ProcessGuard: `domains/infrastructure/process_guard.py` — subprocess crash isolation
- ModelServer: `domains/infrastructure/model_server.py` — semaphore, circuit breaker, backends
- TaskQueue: `domains/infrastructure/task_queue.py` — async priority queue
- ServerState: `domains/infrastructure/server_state.py` — AtomicRef, metrics
- Health envelope: `{"status":"success","data":{"model_loaded":true,"model_type":"..."}}`
- `_finish_job()` in training/router.py: sets status + calls mgr.finish()
- Guard stopped before parent preload to avoid OOM
- ProcessGuard `_monitor_loop` snapshots `_worker` under restart_lock
- `_generate_stream_sync()` ALWAYS runs in-process (subprocess crash isolation adds no value for streaming)
- `/chat` uses streaming → `_acquire_model()` → triggers `_get_model()` → parent materializes
- `/inference/generate` uses non-streaming → checks `_use_guard()` first → delegates to guard
- `MAN_AUTO_WORKFLOW=false` disables auto feedback workflow
- `SLO_LAZY_GUARD_AUTOLOAD` controls lazy-guard path
- Server runs on port 8000, Web UI on port 3000
- Stability Gold Standard: 0% crash, ≤1.2x latency degradation, 0% empty, ≤0.30 CV, 100% response rate

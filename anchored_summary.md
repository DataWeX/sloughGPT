# Anchored Summary

## Current Task
No active task. Last commit: be30e120 feat: feature flags system with CI protection

## Session 2026-08-06 (cont. 5) — Contexts + Barrels Test Coverage
- Covered `contexts/` (all 4 files) + 3 small barrel/config surfaces + `lib/controllers.ts` barrel — 8 new test files / 60 tests.
- `ConvSidebarContext.test.tsx` (11) — default state, localStorage read/persist (true/false/ignore non-true), toggles, setters, storage-throw tolerance, throw outside provider, hook API re-render.
- `ChatContext.test.tsx` (9) — render children, all 3 hooks + `useChatContext` throw outside provider, per-context value exposure, combined merge, nested independent consumption.
- `ChatToolbarContext.test.tsx` (5) — throw outside provider, all 9 groups' values, group callback invocation, live value update via harness re-render.
- `ModelContext.test.tsx` (19) — real `@/hooks/useLiveStatus` store driven via `liveStatusStore` (no mock); `@/lib/model-controller` mocked via `vi.hoisted`. Covers: initial state, no refresh until `ready`, refresh+field mapping (type default `huggingface`, `size_mb→sizeMb`), list failure, loadModel/loadModelPath/unloadModel success+error+backend-error-field, clearError, live-health sync gated on `connectionStatus==='connected'`, `currentModel` from health, `useCurrentModel`/`useModelById`/`useLocalModels`/`useHuggingFaceModels`.
- `compare-config.test.ts` (6) — METRIC_COLUMNS 6 columns, keys on `BenchmarkResult`, fmt/accessor per column, lowerBetter set, Infinity fallback for missing p95/p99.
- `NavIcons.test.tsx` (3) — all 28 exports callable, `IconClose === IconX`.
- `controllers-barrel.test.ts` (4) — identity of 19 controller exports + download helpers vs source modules; `vmController.run/builtins` functions.
- `chat-barrel.test.ts` (3) — identity of 22 chat component exports; `ChatScreen` is memo/forwardRef (object, not function).
- `vitest.config.ts`: `contexts/**/*.test.{ts,tsx}` added to `include` AND jsdom `environmentMatchGlobs`.
- Full suite: 225 files / 2286 tests all pass; tsc exit 0.

## Session 2026-08-06 (cont. 4) — WebGPU Test Coverage
- Covered `lib/soulnet-webgpu/*` with 5 test files / 40 tests: `weights.test.ts` (11, v3 binary/v2 JSON round-trips, `inferArch`, `guessShapes`), `cache.test.ts` (6, fake IndexedDB), `engine.test.ts` (9, real hybrid CPU/GPU forward via fake `GPUDevice`), `transformer-engine.test.ts` (8), `worker.test.ts` (7, `SoulEngineWorker` + real `worker.ts` protocol).
- Config: jsdom `environmentMatchGlobs` entry for `lib/soulnet-webgpu/**/*.test.ts`; helper gained `makeWebGPU` (fake device with queue + MAP_READ readback), `stubWorker`, IDB installer, `.sou` builders.
- Root-cause fixes: v3 `.sou` builder omitted the 4-byte alignment padding before float data that `parseSou` requires (weights.ts:195 `while (offset % 4) offset++`); worker protocol tests needed `vi.resetModules()` for the 2nd dynamic import and token emission *after* `gen.next()`; fake pipeline needed `getBindGroupLayout`; non-JSON test buffer was 5 bytes for an 8-byte string; LSTM nl=2 param indices corrected.
- Full suite: 214 files / 2168 tests all pass; tsc exit 0.

## Session 2026-08-06 — Monitoring Expansion + Settings Crash Fix
- `/monitoring` complete: SSE-driven cards, trend-history recording (`record_trend_snapshots`, 5s throttle), resilience fallback with `mapDetailedToSnapshot` — fixed a `num_parameters: null` data-loss bug. Full frontend suite 1934 tests green, tsc exit 0.
- `/settings` production TypeError root cause: legacy persisted `man-store` replaces whole `settings` object (zustand shallow merge) → `defaultTemp` etc. `undefined`; `?? 0` guard was stripped by SWC (TS `number` type). Fixed: deep-merge in persist `merge` + `SettingsSlider` prop typed `number | undefined`. New `store.migration.test.ts` (3 tests); 462 lib/app tests green, tsc exit 0.

## Session 2026-08-06 (cont.) — Monitoring Test Completion + DPOCard Type Fix
- Closed monitoring coverage gap: 10 new test files (LatencyCard, AlertPanel, ResourceCard, KnowledgeCard, AutoTrainCard, QualityCard, FeedbackCard, TrainingHistory, ExecutorPool, KvCacheCard) = 76 tests. Every monitoring component now has a test file.
- Completed pre-existing in-progress `DPOCard.tsx` typing change (`any` → `Record<string, unknown>`) with `typeof` guards — 5 tsc errors fixed, rendering preserved.
- Full suite: 198 files / 2013 tests all pass; tsc exit 0.

## Session 2026-08-06 (cont. 3) — Query Layer Test Coverage
- Covered the custom query system (`lib/query/*`) with 3 test files / 35 tests:
  - `lib/query/client.test.ts` (16) — `serializeKey`, `fetchQuery` cache hit/dedup-in-flight/retry-success/retry-exhausted/error-storage/fetchingKeys, `invalidateQuery`, `isStale`/`getQueryState`, `subscribeQuery` GC (delete-after-unsub, keep-while-subscribed, cancel-on-resubscribe via fake timers)
  - `lib/query/hooks.test.ts` (13) — `useQuery` mount fetch/enabled=false skips/onSuccess/onError/refetch/refetch-after-invalidation, `useMutation` success+onSuccess/error+onError+onSettled/invalidateKeys/mutate-swallows/reset, `useInvalidate`, `useIsFetching`
  - `lib/query/api-hooks.test.ts` (6) — `@/lib/model-controller` + `@/lib/souls-controller` mocked via `vi.hoisted`; `useModels`/`useSouls`/`useCurrentSoul`/`useCheckpoints` fetchers, `useLoadModel` invalidates `models`, `useSwitchSoul` invalidates souls/current-soul/checkpoints
- `vitest.config.ts`: jsdom `environmentMatchGlobs` entries added for `lib/query/hooks.test.ts` + `lib/query/api-hooks.test.ts`.
- Lesson: `useMutation(fn)` infers `V = void` — passing a string to `mutateAsync` needs explicit `useMutation<string, string>`.
- Full suite: 207 files / 2092 tests all pass; tsc exit 0.

## Session 2026-08-06 (cont. 2) — Remaining Test Coverage Gap
- 6 new test files (48 tests): `lib/sse-client` (9, with reconnect/abort/chunk-buffering), `lib/conversations-utils` (11), `lib/download-utils` (6, DOM via new environmentMatchGlobs entries), `lib/reaction-store` (10), `hooks/useApiHealth` (2), `components/DatasetInlineImportModal` (10).
- Lesson: `vi.mock` factories run at module instantiation → module-level `const`s referenced from them are TDZ; use `vi.hoisted` or state defined inside the factory. jsdom-created anchors are never in the DOM → capture via `createElement` spy.
- Full suite: 204 files / 2057 tests all pass; tsc exit 0.

## Session 2026-07-13 — Server-Side Training from Inference Logs

### Problem
Mobile→server round-trip wasted bandwidth: mobile collected pairs and sent them back, but the server already had every (user_msg, assistant_msg) pair in its session/response logs.

### Solution
Server trains directly from its own inference logs. Two approaches:
1. **On-demand**: Mobile calls `POST /mobile/train/from-sessions` — server extracts pairs from session JSON + response log JSONL, trains, returns checkpoint.
2. **Background auto-train**: `AutoTrainer` monitors new conversations, triggers training when threshold (10) reached.

### Files Created
- `packages/core-py/domains/training/pair_extractor.py` — `extract_pairs_from_sessions()`, `extract_pairs_from_logs()`, `write_training_text()`, `count_pairs_in_sessions()`, `count_pairs_in_logs()`
- `packages/core-py/domains/training/auto_trainer.py` — `AutoTrainer` (background thread, threshold, interval, subprocess spawn), `start_auto_trainer_if_enabled()`, `stop_auto_trainer()`
- `packages/core-py/tests/test_pair_extractor.py` — 21 tests
- `packages/core-py/tests/test_auto_trainer.py` — 17 tests

### Files Modified
- `apps/api/server/routers/mobile.py` — Added `POST /mobile/train/from-sessions` + `GET /mobile/train/auto-status` endpoints
- `apps/api/server/main.py` — Auto-trainer wired into lifespan (start/stop)
- `apps/mobile/src/services/api-client.ts` — `trainFromSessions()`, `getAutoTrainStatus()`
- `apps/mobile/src/services/training-collector.ts` — `trainFromSessions()`, `getAutoTrainStatus()`
- `tests/server/test_mobile_training.py` — 4 new endpoint tests (10→15 total)

### Endpoints
| Endpoint | Purpose |
|----------|---------|
| `POST /mobile/train/from-sessions` | Train from server's own inference logs (no mobile data needed) |
| `GET /mobile/train/auto-status` | Auto-trainer status + config |

### Env Vars
| Var | Default | Purpose |
|-----|---------|---------|
| `MAN_AUTO_TRAIN` | `0` | Enable background auto-training |
| `MAN_AUTO_TRAIN_THRESHOLD` | `10` | Conversations before trigger |
| `MAN_AUTO_TRAIN_INTERVAL` | `300` | Min seconds between trains |

### Test Results
- 21 pair_extractor: ✅
- 17 auto_trainer: ✅
- 15 mobile training endpoints: ✅
- 15 mobile training-collector: ✅
- 514 mobile JS: ✅
- **Total: 68 new tests, all pass**

## Session 2026-07-13 — On-Device Training Pipeline + Storage Integration

### Removed
- **Sensor collection** — Deleted `routers/activity.py`, `test_activity_router.py`, `__mocks__/react-native-sensors.ts`, `__mocks__/rxjs.ts`; removed `react-native-sensors` + `rxjs` from `package.json`; removed Activity Recognition types from `types/index.ts`

### Built — On-Device Training (server-assisted)
1. **training-collector.ts** (mobile) — Collects (user_msg, assistant_msg, quality) pairs, persists to Zustand store, batch-sends to server via `POST /mobile/train`, pulls updated weights back.
2. **POST /mobile/train** (server) — Receives training pairs, stores in MogDB, runs HFFineTuner (GPT-2 + LoRA rank 8, 1 epoch), returns checkpoint name + loss + steps.
3. **Chat auto-collection** — `chat-store.ts` calls `collectPair()` after every successful local or remote response.
4. **Training data CRUD** — Full MogDB-backed CRUD: stats, pending, session lookup, quality update, delete, compact.

### Storage Architecture
```
Mobile: Zustand store (training-data-store.ts) → localStorage persistence
   ↓ (batch send)
Server: MogDB (mobile_training_store.py) → HFFineTuner → checkpoint
```

### Server Endpoints
- `POST /mobile/train` — Store pairs + trigger training
- `GET /mobile/train/stats` — Total/pending/synced/used + quality breakdown
- `GET /mobile/train/pending` — Unsynced pairs (paginated)
- `GET /mobile/train/session/{id}` — Pairs by session
- `PATCH /mobile/train/pair/{id}` — Update quality
- `DELETE /mobile/train/pair/{id}` — Delete single pair
- `DELETE /mobile/train/synced` — Delete all synced pairs
- `POST /mobile/train/compact` — Compact MogDB journals

### Files Created
- `apps/mobile/src/services/training-collector.ts` — Training data collector (Zustand-backed)
- `apps/mobile/src/stores/training-data-store.ts` — Zustand store for training pairs
- `apps/mobile/src/stores/__tests__/training-data-store.test.ts` — 15 tests
- `apps/mobile/src/services/__tests__/training-collector.test.ts` — 15 tests
- `packages/core-py/domains/training/mobile_training_store.py` — MogDB store
- `packages/core-py/tests/test_mobile_training_store.py` — 23 tests
- `tests/server/test_mobile_training.py` — 10 endpoint tests
- `scripts/migrate_training_data.py` — Migrated 732 pairs from training.db → MogDB

### Files Modified
- `apps/api/server/routers/mobile.py` — Added 7 CRUD endpoints + quality_breakdown
- `apps/mobile/src/services/training-collector.ts` — Migrated from AsyncStorage to Zustand
- `apps/mobile/src/services/api-client.ts` — Added `mobileTrain()` + `pullWeights()`
- `apps/mobile/src/stores/chat-store.ts` — Added `collectPair()` calls
- `packages/core-py/pytest.ini` — Added `pythonpath = . ../mogdb/src`

### Test Results
- 10 server endpoint tests: **all pass**
- 23 MogDB store tests: **all pass**
- 15 mobile store tests: **all pass**
- 15 mobile collector tests: **all pass**
- 43 server integration tests: **all pass**
- **Total: 106 training-related tests, all pass**

## Project State (v0.3.0)
- **Inference**: SloNet autograd, SloTransformer, GPT-2/LLaMA import, forward_fast, KV cache, generate_numpy (8.2ms/tok), .slnc mmap, .sou binary
- **Training**: Char LSTM, distill GPT-2→SloTransformer, HF fine-tune+LoRA, auto-train SSE, online LoRA, distributed DDP, **on-device training (server-assisted)**
- **Serving**: ModelServer (semaphore, circuit breaker, MPS OOM recovery, ONNX backend), ModelRegistry, ProcessGuard, torch.inference_mode, CPU thread optimization
- **API**: 20+ FastAPI routers, SSE standard envelope, 31+ CLI commands, shell REPL (40+ commands, pipelines, tab completion)
- **Frontend**: Next.js 25+ pages, Strui component library, 2250+ vitest tests (223 files, all pass), 6 Cypress E2E specs
- **Mobile**: React Native 11 screens, 22 services, on-device inference (JS SloNet + native Metal), **on-device training (server-assisted, MogDB + Zustand stores)**, offline cache
- **Quantization**: int8/int4 AVX2 GEMM kernels, SloLinear quantized forward, SLNC persistence, frontend QuantizationCard
- **Infrastructure**: Rate limiting (sliding window), Prometheus metrics (/metrics + /metrics/prometheus), flash attention config, ONNX backend integration

## Roadmap Status
- **Phase 1-5**: ✅ Complete
- **Phase 6**: ✅ Complete (all 20 items done)
- **Phase 7**: ✅ Complete (React Native + on-device inference + on-device training + offline)

## Key Files
- `ROADMAP.md` — structured task list with priorities and estimates
- `AGENTS.md` — project conventions, commands, architecture
- `apps/mobile/src/services/training-collector.ts` — On-device training data collection (Zustand-backed)
- `apps/mobile/src/stores/training-data-store.ts` — Zustand store for training pairs (indexed, localStorage persistence)
- `apps/mobile/src/services/onnx-inference-service.ts` — JS SloNet runtime (inference-only)
- `apps/mobile/src/services/llama-rn-service.ts` — Native Metal inference via llama.rn
- `apps/mobile/src/stores/chat-store.ts` — Chat flow with auto-training-pair collection
- `apps/api/server/routers/mobile.py` — Mobile BFF + POST /mobile/train endpoint (MogDB-backed)
- `packages/core-py/domains/training/mobile_training_store.py` — MogDB training data store (server-side)
- `packages/core-py/domains/training/mobile_training_store.py` — MogDB training data store (server-side)
- `packages/core-py/domains/training/hf_finetune.py` — HFFineTuner (server-side training)
- `packages/core-py/domains/infrastructure/rate_limiter.py` — RateLimiter + RateLimitMiddleware
- `packages/core-py/domains/infrastructure/metrics.py` — MetricsCollector (Prometheus)
- `packages/core-py/domains/training/distributed.py` — DistributedTrainer wrapper

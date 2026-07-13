# Anchored Summary

## Current Task
No active task. Last commit: 5bf3ed8 perf: SloNet kernel integration + generate optimization + gradient fixes

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
- **Frontend**: Next.js 25+ pages, Strui component library, 2000+ vitest tests, 6 Cypress E2E specs
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

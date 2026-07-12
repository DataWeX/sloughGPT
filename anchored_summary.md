# Anchored Summary

## Current Task
No active task. Phase 6 fully complete. Phase 7 (mobile) pending decision.

## Session 2026-07-12 — Phase 6 Completion Blitz
Completed 7 items in one session:

1. **ONNX backend** — Fixed `_export_to_onnx` to use real `torch` instead of stub; `ONNXBackend` wired into `ModelServer` with priority chain; 6 tests
2. **Flash attention** — Added `use_flash_attention` config option; `_resolve_attn_kwargs()` auto-detects CUDA + SDPA; 2 tests
3. **Interactive model selector** — Curses-based fuzzy search UI (`model select` command); fetches HF + local models; 7 tests
4. **RAG pipeline** — Marked ✅ in roadmap (already fully integrated: `enrich_with_knowledge` → `ContextCore` → system prompt injection)
5. **Rate limiting** — `RateLimiter` sliding window per IP + `RateLimitMiddleware` (BaseHTTPMiddleware); 8 tests
6. **Prometheus metrics** — `MetricsCollector` (counters, histograms, gauges) + `MetricsMiddleware` + `/metrics` + `/metrics/prometheus`; 11 tests
7. **Distributed training** — `DistributedTrainer` wrapper (DDP, gradient accumulation, checkpoint sync); 11 tests

### New Files
- `packages/core-py/domains/infrastructure/rate_limiter.py` — RateLimiter + RateLimitMiddleware
- `packages/core-py/domains/infrastructure/metrics.py` — MetricsCollector (Prometheus text format)
- `packages/core-py/domains/training/distributed.py` — DistributedTrainer + init_distributed/cleanup_distributed
- `packages/core-py/tests/test_onnx_backend.py` — 6 tests
- `packages/core-py/tests/test_rate_limiter.py` — 8 tests
- `packages/core-py/tests/test_metrics.py` — 11 tests
- `packages/core-py/tests/test_distributed.py` — 11 tests
- `apps/cli/tests/test_model_selector.py` — 7 tests

### Modified Files
- `packages/core-py/domains/inference/onnx_engine.py` — Fixed stub torch import to real torch
- `packages/core-py/domains/infrastructure/config.py` — Added `use_flash_attention` to ModelConfig
- `packages/core-py/domains/infrastructure/model_loader.py` — Added `_resolve_attn_kwargs()` for flash attention
- `packages/core-py/domains/infrastructure/model_server.py` — Added `ONNXBackend` class + `onnx_engine` param
- `packages/core-py/domains/models/provider.py` — Added metrics recording to `HFModelProvider.chat()`
- `apps/api/server/routers/metrics.py` — Rewrote to use MetricsCollector
- `apps/api/server/infrastructure/middleware.py` — Added `MetricsMiddleware` to middleware chain
- `apps/cli/src/commands/models.py` — Added `_cmd_models_select` (curses fuzzy selector) + `model select` subcommand
- `ROADMAP.md` — All Phase 6 items marked ✅

### Test Results
- 58 tests across 6 new test files: **all pass**

## Project State (v0.3.0)
- **Inference**: SloNet autograd, SloTransformer, GPT-2/LLaMA import, forward_fast, KV cache, generate_numpy (8.2ms/tok), .slnc mmap, .sou binary
- **Training**: Char LSTM, distill GPT-2→SloTransformer, HF fine-tune+LoRA, auto-train SSE, online LoRA, activity classifier CNN, distributed DDP
- **Serving**: ModelServer (semaphore, circuit breaker, MPS OOM recovery, ONNX backend), ModelRegistry, ProcessGuard, torch.inference_mode, CPU thread optimization
- **API**: 20+ FastAPI routers, SSE standard envelope, 31+ CLI commands, shell REPL (40+ commands, pipelines, tab completion)
- **Frontend**: Next.js 25+ pages, Strui component library, 2000+ vitest tests, 6 Cypress E2E specs
- **Quantization**: int8/int4 AVX2 GEMM kernels, SloLinear quantized forward, SLNC persistence, frontend QuantizationCard
- **Infrastructure**: Rate limiting (sliding window), Prometheus metrics (/metrics + /metrics/prometheus), flash attention config, ONNX backend integration

## Roadmap Status
- **Phase 1-5**: ✅ Complete
- **Phase 6**: ✅ Complete (all 20 items done)
- **Phase 7**: Pending (React Native 8h, CoreML/Metal 16h, Sensors 6h, Offline 4h)

## Key Files
- `ROADMAP.md` — structured task list with priorities and estimates
- `AGENTS.md` — project conventions, commands, architecture
- `packages/core-py/domains/infrastructure/rate_limiter.py` — RateLimiter + RateLimitMiddleware
- `packages/core-py/domains/infrastructure/metrics.py` — MetricsCollector (Prometheus)
- `packages/core-py/domains/training/distributed.py` — DistributedTrainer wrapper
- `apps/cli/src/commands/models.py` — Interactive model selector (curses)
- `apps/api/server/routers/metrics.py` — /metrics + /metrics/prometheus endpoints
- `packages/core-py/domains/inference/onnx_engine.py` — ONNX Runtime inference
- `packages/core-py/domains/infrastructure/model_server.py` — ModelServer + ONNXBackend

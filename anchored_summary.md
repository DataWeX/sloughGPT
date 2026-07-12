# Anchored Summary

## Current Task
No active task. Last commit: 3469b35 fix: broken model_loader imports + _resolve_device consolidation

No active task. All prior work committed and released as v0.3.0.

## Session 2026-07-12 — Distill Fix, Tests, Roadmap, Release
- Fixed CLI API mode `distill` command: `data.get("id")` → `data.get("job_id")` (progress polling was broken)
- Added 36 new tests: `test_distill_gpt2.py` (26 tests for DistillConfig, TextDataset, softmax, KL/CE losses) + `test_training_distill.py` (10 tests for endpoint schema, route, errors, queued status)
- Committed prior session leftovers: pad token fix, provider registration, dead ONNX removal, inference_mode optimization, daemon subprocess fix, SOU ndim+shape format
- Created `ROADMAP.md` — structured phases 1-7 with prioritized tasks
- Bumped version to v0.3.0, tagged release

## Project State (v0.3.0)
- **Inference**: SloNet autograd, SloTransformer, GPT-2/LLaMA import, forward_fast, KV cache, generate_numpy (8.2ms/tok), .slnc mmap, .sou binary
- **Training**: Char LSTM, distill GPT-2→SloTransformer, HF fine-tune+LoRA, auto-train SSE, online LoRA, activity classifier CNN
- **Serving**: ModelServer (semaphore, circuit breaker, MPS OOM recovery), ModelRegistry, ProcessGuard subprocess isolation, torch.inference_mode, CPU thread optimization
- **API**: 20+ FastAPI routers, SSE standard envelope, 31+ CLI commands, shell REPL (40+ commands, pipelines, tab completion)
- **Frontend**: Next.js 25+ pages, Strui component library, 2000+ vitest tests, 6 Cypress E2E specs
- **Quantization**: int8/int4 AVX2 GEMM kernels, SloLinear quantized forward, SLNC persistence, frontend QuantizationCard

## Roadmap
See `ROADMAP.md` for structured task list (Phase 6: active work, Phase 7: mobile).

## Key Files
- `ROADMAP.md` — structured task list with priorities and estimates
- `AGENTS.md` — project conventions, commands, architecture
- `packages/core-py/domains/training/distill_gpt2.py` — GPT-2→SloTransformer distillation
- `apps/cli/src/commands/train.py` — CLI train/distill commands
- `packages/core-py/domains/shell/repl.py` — Shell REPL with train/distill/follow
- `packages/core-py/tests/test_distill_gpt2.py` — 26 distill module tests
- `tests/test_training_distill.py` — 10 distill endpoint tests

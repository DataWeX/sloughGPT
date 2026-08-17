# Changelog

All notable changes to SloughGPT are documented here.

## [0.3.0] - 2026-08-16

### Added

#### Phase 1 — Core Inference Engine
- SloNet autograd system: pure NumPy `Tensor` class with 25+ ops, full forward and reverse-mode AD, bidirectional DAG for JVP/tangent propagation
- SloTransformer architecture: GPT-2 and LLaMA attention blocks, FFN, RoPE, GQA, layer norm, RMS norm
- GPT-2 weight import via universal `build_arch()` converter with auto-detection
- `forward_fast()` + `pre_extract_weights()` for 96x faster inference over naive path
- KV cache for greedy and beam-search generation with per-layer K/V caching
- `generate_numpy()` / `generate_numpy_stream()` — token-by-token streaming in pure NumPy (~5 tok/s on CPU)
- `.slnc` memory-mapped weight format — 2.2x faster load, zero-copy demand paging
- `.sou` checkpoint format (v3 binary) — 1960x faster than JSON, soul metadata embedded
- `MorphTokenizer` — pure Python BPE with morphological analysis, no Rust binary dependency

#### Phase 2 — Training Pipeline
- `SloughGPTTrainer` — char-level and token-level LSTM training from scratch
- GPT-2 → SloTransformer distillation pipeline (`distill_gpt2.py` module + CLI + API endpoint)
- HuggingFace fine-tuning via `transformers.Trainer` with optional LoRA (`peft`)
- Auto-train SSE streaming with `TrainingSequence` phases (GENERATE_DATA → DISTILL → TRAIN → EVALUATE → DEPLOY → COMPLETE)
- Online LoRA adapter updates driven by user feedback
- Activity classifier — CNN on sensor data, 87.5% validation accuracy
- Gradient clipping + `SloReduceLROnPlateau` LR scheduler
- Data augmentation (Gaussian noise, amplitude scaling, time shift, channel dropout)
- `BaseTrainer` protocol + `TrainResult` dataclass for standardized training interface

#### Phase 3 — Model Serving
- `ModelServer` — asyncio semaphore, configurable timeout, pre/post-generation hooks, circuit breaker (3 failures → 30s open), MPS OOM recovery
- `ModelRegistry` — composable registry with TTL cache, health summary, per-model metrics
- `ProcessGuard` — subprocess isolation with memory tracking, crash recovery, streaming delegation
- Read/write separation: tokenize with read semaphore, generate with write semaphore
- Health endpoint with detailed metrics (`/health`, `/health/detailed`)
- Session-level KV cache for multi-turn conversation speedup
- Cross-turn cache invalidation and re-use

#### Phase 4 — API & CLI
- FastAPI routers with 20+ endpoints (chat, generate, models, training, souls, feedback, knowledge, agents, datasets, multimodal, system)
- Standard SSE envelope (`{stream, phase, status, data, meta, message}`)
- Python CLI with 31+ Click-based commands (model, train, checkpoint, distill, shell, etc.)
- Interactive Shell REPL with 40+ commands, pipelines (`|`), background jobs (`&`), tab completion, env vars, aliases, pager, reverse history search
- Shell onboarding tutorial and `~/.config/sloughgpt/rc` startup file

#### Phase 5 — Frontend
- Next.js app with 32 pages (chat, models, training, settings, monitoring, datasets, agents, knowledge, compare, export, tokenizer, labs, etc.)
- `@sloughgpt/strui` component library (Card, Button, Input, Dialog, Dropdown, Toast, Chat components, 50+ icons)
- Chat with markdown rendering, code blocks, streaming, regeneration, feedback (thumbs up/down), knowledge injection, image upload, voice input
- Training page with distill + fine-tune paths, loss chart, checkpoint catalog
- Model catalog with soul switcher, thumbnails, model details dialog, unload
- 15+ axios-based controllers (migrated from legacy `api.ts`)
- Knowledge management (CRUD, batch, search, categories, JSON export/import)
- System health monitoring with real-time CPU/memory chart
- Keyboard shortcuts modal, global shortcuts (`Ctrl+1-5`, `Ctrl+N`, `?`)
- E2E test suite (Cypress): 6 specs, 12 assertions
- 2000+ vitest tests covering all lib files

#### Phase 6 — Inference Quality, Training, Model Management, RAG, CLI, Infrastructure
- Beam search with KV cache, repetition penalty fix, top-k sampling in SloNet numpy
- Flash attention for HF models
- Distill eval metrics (perplexity, BLEU), checkpoint resume from `.soul`
- LoRA merge export (`.sou`), distributed training
- Model quantization UI (INT8/INT4), auto-download from HuggingFace Hub
- Model comparison benchmark (side-by-side quality metrics)
- SLNC auto-conversion on load
- RAG pipeline integration with document chunking strategies
- Knowledge auto-ingest from chat conversations
- ONNX backend wired into ModelServer
- Docker deployment (`Dockerfile` + `docker-compose`)
- Rate limiting, Prometheus metrics
- Shared `TrainingExecutor` with thread pool, job tracking, cancellation, PGQ integration
- Executor API endpoints + frontend monitoring card

#### Phase 7 — Mobile
- React Native app shell with 11 screens, 22 services, navigation
- On-device inference via CoreML/Metal (`onnx-inference-service.ts` + `llama-rn-service.ts`)
- On-device training with server-assisted pipeline
- Offline mode with message cache and pending sends

### Changed
- **PyTorch removed as a hard dependency** — SloNet (pure NumPy) handles all training and inference; PyTorch only imported lazily for optional ONNX/TorchScript export and HF fine-tune
- Default inference device forced to CPU on Intel Macs (MPS false-positive fix for PyTorch 2.x on x86_64)
- Model catalog now queries HuggingFace Hub API for real model listings (was 9 hardcoded IDs, now 50+)
- Watchdog recovery changed to log-only (eliminates model reload from background thread)
- `sentence-transformers` removed — replaced by n-gram TF-IDF embedder
- `bitsandbytes` removed — replaced by `Quantine` (pure NumPy quantization)
- Frontend API layer fully migrated from single `api.ts` (2100 lines) to per-domain controllers
- All SSE endpoints standardized to `{stream, phase, status, data, meta, message}` envelope
- `state.py` rewritten to delegate module-level access to `AtomicRef` instances for thread safety

### Fixed
- Cross-attention gradient explosion (800x) caused by redundant `SloLayerNorm` in `SloCrossAttention`
- `generate()` non-determinism from Apple Metal GPU accelerator — accelerator disabled during inference for deterministic results
- MPS OOM crashes after ~10 sequential requests — forced CPU inference with session-level KV cache
- Streaming chat hang caused by synchronous blocking call (`_enrich_knowledge`) inside async generator — offloaded to thread pool
- Model loading provider update — new model now immediately takes effect in chat
- Event loop corruption from unawaited coroutine in `_auto_ingest()`
- KV cache indexing bug in optimizer (hardcoded `layer_idx=0`)
- Settings crash in production — Zustand persist shallow-merge overwriting missing fields with `undefined`
- 33 frontend test failures (JSDOM StrictMode double-mount, Radix Dialog portals, fetch mock timing)
- 14 flaky frontend tests (DOM timing, async renders)
- Deprecated `datetime.utcnow()` calls replaced with `datetime.now(timezone.utc)` across 7 files

### Removed
- `SloNetProvider` class (130 lines, zero consumers)
- `api.ts` legacy file (2100 lines) — all consumers migrated to individual controllers
- Dead enterprise, UI, and integration domain packages (~1770 lines)
- Dead hooks (`useIndexedDBSessions`, `useModelContext`, `useModelLoader`, `useStatus`, `useStreamingChat`, `chat-reveal`)
- Duplicate source files from `apps/web/components/ui/` (25 files — components now live only in `packages/strui`)
- Stale redirect pages (`/api-docs`, `/experiments`, `/plugins`, `/recents`)
- 24 dead inline `@app` endpoints shadowed by router registrations

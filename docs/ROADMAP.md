# sloughGPT roadmap

> Direction: a general-AI **model**, not a net. Owned architecture (SloNet),
> owned objective (beyond next-token prediction), owned data. Every item serves
> all four domains — **Cognitive, Core, Gateway, Training** — with the
> serving engine (`InferenceEngine` + `SloNetServer`) hardened first and the
> gateway built well as the edge optimizer of the core infra stack.
> Borrowed weights (HF conversion, distillation) are bootstrap, not destination.

## Current state (September 2026)

### Model — owned machinery, bootstrap weights

- Machinery is owned: SloNet pure-NumPy autograd (full DAG, 25 ops), SloTransformer, TokenTree tokenizer, `.soul`/`.slnc` formats with mmap loading. `NativeEngine` (RoPE/RMSNorm/GQA/KV-cache/top-p/top-k) loads real `.slnc` weights with the matching real tokenizer, wired via `setup_providers(native_slnc_path=...)` (opt-in).
- Weights are still borrowed (Qwen2.5-0.5B-Instruct via CPU; distillation GPT2 → LSTM; HF fine-tuning + LoRA) — bootstrap, not destination. Fully-owned loop (own init → own data → own objective) exists only as `SloChatTrainer` on chat pairs.

### Engine (hardened first)

- `SloNetServer` + `ModelServer`/`ModelRegistry`/`CircuitBreaker`, warmup, metrics. ProcessGuard isolation is the production default (manual + autoload paths, runtime toggle). Lazy header-only `.slnc` provider defers weight pages. Standalone `InferenceEngine` over length-prefixed TCP (`InferenceClient`), so inference work is isolated from the API process.

### Four domains

- **Cognitive** — consciousness engine post-processes generation (`ConsciousnessConfig.level`); Personality/Memory/Style/Task context managers; auto-memory service learning from conversations; souls/traits with feedback-driven weight updates.
- **Core** — provider registry + router with processor pipeline (Vision/Knowledge/ToolUse/ Personality/Style), session core + KV-cache reuse, priority queue with streaming slot reservation, cancel manager, persistent settings. Serves chat/stream/regenerate (standardized SSE envelope) plus the raw `/inference/generate` path and the `[[TOOL: name]]` tool loop.
- **Gateway** — Rust edge builds clean with rustls (`slough-gateway`, binds `:8080`, health + degraded-sidecar behavior verified) but is bypassed: frontend points at `:8000` direct after the model-aware proxy proved to be dead infra. Remaining work is the thin generic edge (byte-relay, filters, chat-only) — tracked as Gateway-domain work, not plumbing.
- **Training** — single composable `TrainingLoop` (`training_handler.py`: samplers, gradient handlers, loss trackers, checkpoint savers) replacing the three loop copies; distillation, LoRA (including background feedback loop), activity classifier.

### Clients (separate from core infra — see PRODUCT_ENGINEERING.md)

- Web: 32 pages on `@sloughgpt/strui`, per-domain controllers over one HTTP client, markdown chat with streaming/regeneration/feedback. CLI: Python REPL (40+ commands) + opt-in curses TUI. Voyager journey library (201 tests, 7 backends). Mobile: not started. Clients reach core only through the API/protocol — never internals.

### Compression (pugqeep)

- VQ cluster path is wired end to end: k-means++ init + entropy-adaptive k + Lloyd's early-stop refinement, Huffman-coded assignments, `Point.generate` decodes on read. `test_pugqeep_compressor.py` 159/159 green (cluster/function/round-trip/serialization/nbytes + extended suites). Open: Lloyd+Huffman vs Q4-block benchmark, plus a dedicated Huffman round-trip test (decode is currently covered only implicitly via decompress round-trips).

### What the bootstrap era taught us (keep the lessons, drop the habits)

- Borrowed weights earned their keep: HF conversion + distillation got real inference running years before owned weights could. Keep as bootstrap path; never again as the headline.
- Three training loops grew because each feature trained "just this once" — consolidated into one `TrainingLoop` with adapters. New objectives arrive as adapters, never as fourth loops.
- A model-aware gateway rotted into dead infra (double JSON, buffered streaming, schema drift → rollback to `:8000`). Edge stays generic and thin; model logic lives in Core.
- Routers reaching into `domain.*_internal*` created bypass webs (voice, knowledge, training). Every capability now gets one engine class; routers delegate.
- Prompt middleware (processors) works and stays — but injections are context-token spend, so each one justifies its tokens.
- Feedback→LoRA was wired years late, so learning lagged use. Training story ships with the feature now (fairness rule).

## Near-term goals

1. ~~**Stabilize SloNet training** — Fix remaining backward pass broadcast bugs (test_tokenizer.py failures).~~ **Done** — `_mul` backward uses `_broadcast_back`; `test_slonet_broadcast.py` + `TestSloEngineLearn` 17/17 pass. Remaining: profile and optimize hot loops.
2. ~~**Wire process isolation in production** — Enable ProcessGuard for `_load_hf_model()` so subprocess crashes don't take down the API server.~~ **Done** — guard wiring exists on the manual (`controllers/models.py:_load_hf_model`, lazy + eager) and autoload (`startup.py:_try_lazy_guard_autoload`) paths; `ServerConfig.enable_process_guard` is now the single source of truth and defaults to enabled (`SLO_ENABLE_PROCESS_GUARD`, default `true`), fixing the dead-config mismatch where the field defaulted to `false` while the runtime toggle defaulted to `true`.
3. ~~**Fix pre-existing test failures** — 14 flaky frontend tests (DOM timing, async renders, StrictMode double-mount).~~ **Done** — suite at 324 files / 3048 tests; only `ModelDetailPage.test.tsx` excluded (worker-harness hang).

## Medium-term goals

4. ~~**Incremental training from feedback** — Wire OnlineLoRAUpdater + PerUserLORAStore into a continuous background loop (currently only fires on explicit aggregation).~~ **Done** — `_run_background_training` now reads the tokenizer from the workflow's `set_model()` state (was reading a nonexistent `lora_updater._tokenizer`, so the loop always no-opped); the active server model is wired into the workflow at startup (`main.py:_start_feedback_workflow`) and on feedback (`FeedbackController._wire_model` falls back to `server_state.model` when no auto-train student is set).
5. ~~**Multi-agent orchestration polish** — Async executor works, needs UI for agent creation/editing and dashboard for runs.~~ **Done** — full agent CRUD (create/edit/delete + validation), multi-agent orchestration card (goal/context, per-agent picks via `agent_ids`, live plan→execute→compose→complete SSE timeline), and runs dashboard on `apps/web/app/(app)/agents/page.tsx` (list/timeline views, status + agent filters, expandable detail with per-task dots/previews, result + logs). Backend: `apps/api/server/routers/agents.py` CRUD + `POST /orchestrate` (SSE, `asyncio.gather` level execution) + `GET /runs` + `GET /runs/{run_id}`; persistence via `packages/core-py/domains/agents/run_history.py` (file-backed `data/agent_runs/`). Covered by 26 frontend + 179 backend/core tests.
6. ~~**Dataset management UI** — Import/export/versioning/search frontend.~~ **Done** — list page (search/sort/preview/compare/export/delete/version badges), detail page (rename/stats/quality/insights/preview/snapshots/JSONL+CSV export/convert-to-chat-format), import modals (local/GitHub/HF/URL/ISBN/Kaggle/CSV), chat→dataset export. Backend convert + versioning covered by tests.
7. ~~**Voyager journey testing library** — Build a modular, cross-platform library for user journey tests and computer-use automation.~~ **Done** — `packages/voyager/` with 201 tests, 7 backends (Playwright, Selenium, CDP, API-only, CLI, Appium, Desktop), AI model integration, learning system, and full computer-use primitives (Mouse, Keyboard, InteractionChain, visual detection, smart waits, time-travel debugging, network interception, performance markers).

## New goals (September 2026)

8. ~~**Voyager journey tests for sloughGPT** — Write end-to-end journey tests using Voyager against the sloughGPT frontend.~~ **Done** — 8 journeys (chat, souls, training, datasets, models, knowledge, settings, agents). API backend runner + Playwright runner in `scripts/run_journey_tests.py`. `packages/voyager/tests/test_sloughgpt_journeys.py`.
9. ~~**Wire Consciousness to inference** — Connect `domain/consciousness/` cognitive engine to the inference pipeline.~~ **Done** — `SloEngine._init_consciousness()` loads consciousness engine. `SloEngine.generate()` processes responses through `ConsciousnessEngine.process()` post-generation. Configurable via `ConsciousnessConfig.level`.
10. ~~**Consolidate training loops** — Three copies of the forward/loss/backward/grad-clip/optimize loop exist (`train_pipeline.py`, `chat_trainer.py`, `consciousness/training.py`). Consolidate into `SloughGPTTrainer` with dataset adapters.~~ **Done** — `training_handler.py` is the single composable training engine with protocols (BatchSampler, GradientHandler, LossTracker, CheckpointSaver) and implementations (RandomBlockSampler, PermutationSampler, ChatPairSampler, DirectGradientHandler, AccumulationGradientHandler, RawLossTracker, EMALossTracker, SoulCheckpointSaver, NpzCheckpointSaver). All three training files now import from `training_handler.py`. One gap: `domain/consciousness/training.py` was deleted in the layout refactor and never re-created, so the router's `_get_trainer()` still imports `domain.consciousness.training` and 500s on `/status` and `/train/*`. Restored as a dataset adapter in goal 15.
11. ~~**Voice router cleanup** — TTS works, phoneme encode works, but router bypasses domain. Wire `domain/voice/` properly to `apps/api/server/routers/voice.py`.~~ **Done** — Both `voice.py` and `phoneme.py` routers properly delegate to `domain.voice` via `get_voice_engine()` and `get_phoneme_engine()`. No `_internal` imports.

## Training delivery — two tracks (October 2026)

The destination is an **owned model** (milestones 12–14): SloNet trains from scratch on its own data and objective. A years-long "couldn't train" streak (broken router import, untested loops, silent divergence risk) proves the failure mode: a months-long run that diverges at hour 30 is a total loss. Delivery therefore splits into two tracks that share the one loop (goals 15–19).

- **Track A — Ship (now).** Fine-tune a small open base (Phi/Qwen/Mistral-1B class) with LoRA on our own data + the consciousness layer, benchmark quality, release. It is client value today and keeps momentum while the owned model cooks. Borrowed weights are bootstrap, not destination (per the lessons above).
- **Track B — Own model (months).** `SloughGPTTrainer` pretraining from scratch as a long unattended job: checkpoint + resume on crash, streaming loss/monitor logs, benchmark hooks. Launch only after the tiny convergence proof (goal 16) is green — never before.

"Can train" is defined operationally: a real run completes and produces a **`TrainingStatus`/`TrainResult`** (loss, epoch, pairs, duration) that tests assert on and benchmarks record. Everything below hangs off that.

15. **Train status contract is real** — `/consciousness/status`, `/consciousness/train/status`, `/self-train/status`, `/training/start` all report a testable status from the one loop. Restore `domain/consciousness/training.py` as a thin dataset adapter over `training_handler.py` (fixes the dead import, no new loop). Status transitions asserted in backend tests; `scripts/benchmark_slonet_training.py` records loss and tokens/sec.
16. **Tiny convergence proof** — the one loop drives a small model on a small corpus to near-zero loss in <100 steps, asserted in test + benchmark. This is the release gate for *any* longer run: no long job starts until the loop has demonstrably converged once (catches gradient/layernorm bugs before they eat a month).
17. **Track A ship — trainable fine-tune** — LoRA (`HFLoraTrainer`) on a small open base with our data; benchmark reports quality + speed; click-through journey (dataset → start → loss curve → checkpoint → load into chat) passes for a non-technical user. This is the release.
18. **Track B — long-run pretraining** — owned from-scratch run with checkpoint/resume, monitoring, and benchmark hooks; runs for months while Track A ships. Starts only after goal 16 is green.
19. **Scatter gate** — new training surfaces (pages, routers, adapters) are rejected unless they feed the one loop. Adapters only; a fourth loop gets the PR rejected.
20. **Consciousness re-homed under cognition** — move `domain/consciousness/_internal/` modules into `domain/cognitive/_internal/consciousness/` as the self-awareness/metacognition slice of the cognitive domain; clean move, all references (router, tests, consumers) updated in one pass, old files deleted — no deprecated shims. Training stays a dataset adapter into the one loop (goal 15). Branding stays `consciousness` in the public API; only its physical home changes.

## Owned-model milestones (the destination past bootstrap)

12. **Owned architecture** — SloNet stands alone: no HF conversion in the default path, native `.soul` training → `.slnc` serving end to end.
13. **Owned objective** — at least one training objective beyond next-token prediction (memory consolidation, tool-use success, planning) running in the single `TrainingLoop`.
14. **Owned data** — the model trains on experience it generates (chat sessions, tool outcomes, feedback), not only third-party text.

## Deferred (potential Rust)

| Item                         | When                                                               | Why Rust                                        |
| ---------------------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| SloNet kernel rewrite (PyO3) | If training profiling shows Python loop overhead is the bottleneck | 10-50x speedup on backward pass ops             |
| CLI rewrite (binary)         | If startup time or distribution becomes a pain point               | Instant startup, single binary, native readline |
| Token streaming proxy        | If Python async polling becomes a bottleneck                       | Zero-gap streaming, clean cancellation          |

Revisit after stabilizing the Python codebase — current bottlenecks are model inference (2s/request CPU), not Python overhead.

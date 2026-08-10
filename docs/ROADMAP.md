# sloughGPT roadmap

## Current state (July 2026)
- **Inference**: Qwen2.5-0.5B-Instruct via CPU (MPS unstable on 8GB). Chat/streaming/regenerate endpoints stable. SSE envelope standardized.
- **Native C inference engine**: `NativeEngine` (RoPE/RMSNorm/GQA/KV-cache/top-p/top-k) now loads real `.slnc` weights with the matching real tokenizer (`MorphTokenizer.from_pretrained`). `NativeEngine.from_slnc_file()` derives the HF id from the cache path and auto-attaches the tokenizer; chat prompts use the model's real chat template and stop ids. Wired into the provider chain via `setup_providers(native_slnc_path=...)` (opt-in; feature flag still off). Real end-to-end verified on `Qwen/Qwen2.5-0.5B-Instruct` (~12s load, ~1 tok/s CPU, coherent output).
- **Training**: SloNet pure NumPy autograd (full DAG, forward+backward, 25 ops). Distillation pipeline (GPT2 teacher → LSTM student). HuggingFace fine-tuning via `transformers.Trainer` + optional LoRA. Activity classifier (CNN on sensor data).
- **Core**: ModelServer + ModelRegistry + CircuitBreaker for composable serving. ProcessGuard for process-level isolation (wired but not production default). Context managers (Personality/Memory/Style/Task). Feedback workflow (LoRA + eval + aggregation).
- **Frontend**: 32 pages, all migrated to `@sloughgpt/strui` components. Controllers migrated from legacy `api.ts` to per-domain controllers. Chat with markdown, streaming, regeneration, feedback. Full suite: 324 test files / 3048 tests pass, tsc exit 0. `app/(app)/model/[id]/ModelDetailPage.test.tsx` is excluded by design (worker-harness hang, documented in `vitest.config.ts`).
- **CLI**: Python REPL with 40+ commands, pipelines, env, aliases, tab completion, pager (`~/.config/sloughgpt/shell_state.json`). Line-mode shell is the default UI; opt-in split-pane curses TUI (`sloughgpt tui`, `shell --tui`, or `MAN_TUI=1`) with console/output panes, status bar, input row, reverse+forward history search (Ctrl+R/S), output-pane content search (`/`, with `n`/`N` repeat-last-match), readline-style line editing (caret, word movement via Alt+F/B and Ctrl+arrows, transpose Ctrl+T, delete word forward Alt+D, Ctrl+W/U/K/A/E/D, kill ring + Ctrl+Y yank), and Ctrl+C command interrupt — all fixes verified under a real PTY.
- **Mobile**: Not started.

## Near-term goals
1. ~~**Stabilize SloNet training** — Fix remaining backward pass broadcast bugs (test_tokenizer.py failures).~~ **Done** — `_mul` backward uses `_broadcast_back`; `test_slonet_broadcast.py` + `TestSloEngineLearn` 17/17 pass. Remaining: profile and optimize hot loops.
2. **Wire process isolation in production** — Enable ProcessGuard for `_load_hf_model()` so subprocess crashes don't take down the API server.
3. ~~**Fix pre-existing test failures** — 14 flaky frontend tests (DOM timing, async renders, StrictMode double-mount).~~ **Done** — suite at 324 files / 3048 tests; only `ModelDetailPage.test.tsx` excluded (worker-harness hang).

## Medium-term goals
4. **Incremental training from feedback** — Wire OnlineLoRAUpdater + PerUserLORAStore into a continuous background loop (currently only fires on explicit aggregation).
5. **Multi-agent orchestration polish** — Async executor works, needs UI for agent creation/editing and dashboard for runs.
6. **Dataset management UI** — Import/export/versioning/search frontend (backend endpoints exist, frontend stubs).

## Deferred (potential Rust)
| Item | When | Why Rust |
|------|------|----------|
| SloNet kernel rewrite (PyO3) | If training profiling shows Python loop overhead is the bottleneck | 10-50x speedup on backward pass ops |
| CLI rewrite (binary) | If startup time or distribution becomes a pain point | Instant startup, single binary, native readline |
| Token streaming proxy | If Python async polling becomes a bottleneck | Zero-gap streaming, clean cancellation |

Revisit after stabilizing the Python codebase — current bottlenecks are model inference (2s/request CPU), not Python overhead.

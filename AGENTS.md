# Project Rules

## Agent Behavior

### SOP — Work Workflow (follow in order for every task)

1. **Create a todo on kanban** — log the task before starting work.
2. **Branch before implementing** — `git checkout -b feat/<name>` for major implementations (new methods, training loops, inference engines, quantization, model architecture changes). Small fixes, config tweaks, and doc edits can go straight to main.
3. **Do the production workflow** — follow the established dev flow (lint, typecheck, test).
4. **Ask leading questions** — clarify requirements before writing code.
5. **Do work in one shot** — make all edits in a single pass, no back-and-forth.
6. **Check if it's done** — verify the output matches what was asked.
7. **Clarify** — confirm with user if anything is ambiguous.
8. **Ask for more info** — if blocked or unclear, ask before guessing.
9. **Test for bugs or errors** — run lint, typecheck, and tests.
10. **Benchmark** — run the relevant benchmark. If it regresses or fails, fix or delete the branch and revert: `git checkout main -- <files>` or `git branch -D feat/<name>`. Do not merge broken implementations.
11. **Merge when green** — `git checkout main && git merge feat/<name>` only after tests + benchmarks pass.
12. **Submit with a console summary** — commit with a concise summary of what changed.
13. **Build by user journey** — frame work as user journeys (`docs/UX_FLOWS.md`), not files. Walk each journey click-by-click in the running app; build only what blocks it. A journey passes when a non-technical user can complete it.
14. **Summarize, don't dump** — UIs and reports show concise summaries (key metrics, highlighted states), never raw thousand-line dumps. Raw logs stay one click away, collapsed by default.
15. **One global banner** — app-wide alerts go through `useBannerStore` + `<GlobalBanner />` in `AppLayout`, never a per-page banner. Toasts for transient confirmations; banners for journey blockers with actions. Dedupe with `key`.

### Core Rules

- **ALWAYS check if something already exists before building it.** Before creating new files, modules, or features, search the codebase for existing implementations. Use `grep`, `glob`, and `find` to check for existing code, patterns, or similar functionality. Duplicate work wastes time and creates confusion. **This has cost us hours of wasted effort — never skip this step.**
- **NEVER create duplicate docs.** Before creating new documentation, check if the topic is already covered. Update existing docs instead of creating new ones. Only create new docs if the topic is genuinely new and inferentially different from existing structure.
- **NEVER duplicate training loops.** The training infrastructure lives in `domain/training/_internal/`. There is ONE training loop (`SloughGPTTrainer` in `train_pipeline.py`), ONE tokenizer pipeline, ONE checkpoint system. If you need to train something new, create a **dataset adapter** that feeds into the existing `SloughGPTTrainer`, or extend it with a new mode. Do NOT copy-paste the forward/loss/backward/grad-clip/optimize loop into a new file. The three existing copies (`train_pipeline.py`, `chat_trainer.py`, `consciousness/training.py`) are a known debt — we are actively consolidating them. Before writing any training code, ask: "Can I reuse `SloughGPTTrainer` or `SloChatTrainer`?" If the answer is yes, write an adapter, not a new trainer.
- **Do NOT make changes or delete files without explicit user approval first.** Always describe what you plan to do, wait for confirmation, then execute. Even if the user asks you to "build X" or "fix Y", confirm the approach before writing code.
- **During discussions and brainstorming, DO NOT take action.** When the user is explaining ideas, asking questions, or exploring concepts — listen and respond verbally. Do not edit files, create docs, or make changes unless explicitly told to. Jumping ahead during discussion breaks the conversation flow and shows you're not reading the interaction context.
- **Always use the project venv.** Check for `.venv/`, `venv/`, or `poetry env` before running Python commands. Never use bare `python` or `pip` without activating the project environment first.
- **Run benchmarks after every training/inference implementation.** After implementing or modifying any fine-tuning method, training loop, inference optimization, quantization, or model architecture change, run the relevant benchmark before committing. This catches performance regressions and validates improvements. Benchmark scripts live in `scripts/benchmark_*.py`. Key benchmarks:
  - `scripts/benchmark_slonet_training.py` — training speed, convergence, loss curves
  - `scripts/benchmark_quantization.py` — int8/int4 quality vs speed tradeoffs
  - `scripts/benchmark_slonet.py` — inference throughput, latency
  - `scripts/benchmark_stability.py` — generation consistency across runs
  - `scripts/benchmark_latency.py` — end-to-end request latency
  - `scripts/benchmark_results.py record/history/compare` — track results over time
  - If no benchmark exists for your change, create one in `scripts/` and document what it measures.

## File Safety

- **NEVER delete user data files (notes, configs, data stores) without explicit user approval first.** Ask before removing any file that contains user-created content. Propose the change, explain consequences, and wait for confirmation.
- When consolidating or migrating data, always preserve the original as a backup before modifying.
- If a file is not git-tracked, treat it as irreplaceable — ask twice.

## Architecture

- **Notes** (`~/.config/dev-notes/*.md`) are the user's journal — source of truth for task metadata (sprint, gh, status, body).
- **Board** (`.kanban/board.jsonl`) is the kanban view derived from notes via sync.
- Sync is bidirectional: note status ↔ card column.
- **Product docs** (`docs/PRODUCT_ENGINEERING.md`) are the source of truth for what to build. Reference before creating new features, routers, or pages. User flows in `docs/UX_FLOWS.md`, persona in `docs/USER_PERSONA.md`.

## Core Infrastructure Sync Rule

**http-client.ts is the single source of truth for all API communication.**

When backend endpoints, response envelopes, or error formats change:

1. **Update http-client.ts first** — match the new backend contract (endpoints, response shape, error codes).
2. **Propagate to consumers** — update hooks, controllers, and components that use http-client.
3. **Never bypass http-client with raw `fetch()`** — all API calls must go through `apiGet`, `apiPost`, `apiPut`, `apiDelete`, `apiPatch`, or the `request()` function.

This ensures consistent error handling, retries, caching, circuit breaking, and interceptors across the entire frontend. Bypassing http-client breaks these guarantees and creates silent bugs.

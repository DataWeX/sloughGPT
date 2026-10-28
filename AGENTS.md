# Project Rules

## Agent Behavior

### SOP — Work Workflow (follow in order for every task)

1. **Create a todo on kanban** — log the task before starting work.
2. **Branch before implementing** — `git checkout -b feat/<name>` for major implementations (new methods, training loops, inference engines, quantization, model architecture changes). Small fixes, config tweaks, and doc edits can go straight to main.
3. **Do the production workflow** — follow the established dev flow (lint, typecheck, test).
4. **Expert review before implementing** — when the user proposes an approach (design, mechanism, invariant, or architecture idea), do not just implement it. Always respond in the expert-review format:
   - **Verify soundness** — confirm the plan is logically sound, and name the real systems invariant/pattern behind it (cache coherence, MVCC versioning, generation counters, ETags, etc.).
   - **Affirm what's right** — explicitly list the correct instincts in the proposal (e.g., invalidating at the start of a mutation, matching dependent state to its source).
   - **Always give two refinements** — (1) a mechanism/placement correction from a systems lens (where the check lives, push vs. lazy invalidation, coupling boundaries, what gets invalidated when), and (2) a scope correction (what must vs. must not be affected, and what the plan does not yet cover).
   - **Propose concrete implementation** — give step-by-step implementation, then ask for go-ahead before writing code.
5. **Ask leading questions** — clarify requirements before writing code.
6. **Do work in one shot** — make all edits in a single pass, no back-and-forth.
7. **Check if it's done** — verify the output matches what was asked.
8. **Clarify** — confirm with user if anything is ambiguous.
9. **Ask for more info** — if blocked or unclear, ask before guessing.
10. **Test for bugs or errors** — run lint, typecheck, and tests.
11. **Benchmark** — run the relevant benchmark. If it regresses or fails, fix or delete the branch and revert: `git checkout main -- <files>` or `git branch -D feat/<name>`. Do not merge broken implementations.
12. **Merge when green** — `git checkout main && git merge feat/<name>` only after tests + benchmarks pass.
13. **Submit with a console summary** — commit with a concise summary of what changed.
14. **Build by user journey** — frame work as user journeys (`docs/UX_FLOWS.md`), not files. Walk each journey click-by-click in the running app; build only what blocks it. A journey passes when a non-technical user can complete it.
15. **Summarize, don't dump** — UIs and reports show concise summaries (key metrics, highlighted states), never raw thousand-line dumps. Raw logs stay one click away, collapsed by default.
16. **One global banner** — app-wide alerts go through `useBannerStore` + `<GlobalBanner />` in `AppLayout`, never a per-page banner. Toasts for transient confirmations; banners for journey blockers with actions. Dedupe with `key`.

### Core Rules

- **ALWAYS check if something already exists before building it.** Before creating new files, modules, or features, search the codebase for existing implementations. Use `grep`, `glob`, and `find` to check for existing code, patterns, or similar functionality. Duplicate work wastes time and creates confusion. **This has cost us hours of wasted effort — never skip this step.**
- **NEVER create duplicate docs.** Before creating new documentation, check if the topic is already covered. Update existing docs instead of creating new ones. Only create new docs if the topic is genuinely new and inferentially different from existing structure.
- **NEVER duplicate training loops.** The training infrastructure lives in `domain/training/_internal/`. There is ONE training loop (`SloughGPTTrainer` in `train_pipeline.py`), ONE tokenizer pipeline, ONE checkpoint system. If you need to train something new, create a **dataset adapter** that feeds into the existing `SloughGPTTrainer`, or extend it with a new mode. Do NOT copy-paste the forward/loss/backward/grad-clip/optimize loop into a new file. The three existing copies (`train_pipeline.py`, `chat_trainer.py`, `consciousness/training.py`) are a known debt — we are actively consolidating them. Before writing any training code, ask: "Can I reuse `SloughGPTTrainer` or `SloChatTrainer`?" If the answer is yes, write an adapter, not a new trainer.
- **Do NOT make changes or delete files without explicit user approval first.** Always describe what you plan to do, wait for confirmation, then execute. Even if the user asks you to "build X" or "fix Y", confirm the approach before writing code.
- **During discussions and brainstorming, DO NOT take action.** When the user is explaining ideas, asking questions, or exploring concepts — listen and respond verbally. Do not edit files, create docs, or make changes unless explicitly told to. Jumping ahead during discussion breaks the conversation flow and shows you're not reading the interaction context.
- **Always use the project env (conda `sloughgpt`).** Never use bare `python` or `pip` outside the project env. Resolution order lives in `scripts/python`: `SLO_PYTHON` → conda env `sloughgpt` → `.venv` → `python3`; prefix commands with `./scripts/python` or `./run.sh` instead of activating by hand.
- **Run benchmarks after every training/inference implementation.** After implementing or modifying any fine-tuning method, training loop, inference optimization, quantization, or model architecture change, run the relevant benchmark before committing. This catches performance regressions and validates improvements. Benchmark scripts live in `scripts/benchmark_*.py`. Key benchmarks:
  - `scripts/benchmark_slonet_training.py` — training speed, convergence, loss curves
  - `scripts/benchmark_quantization.py` — int8/int4 quality vs speed tradeoffs
  - `scripts/benchmark_slonet.py` — inference throughput, latency
  - `scripts/benchmark_stability.py` — generation consistency across runs
  - `scripts/benchmark_latency.py` — end-to-end request latency
  - `scripts/benchmark_results.py record/history/compare` — track results over time
  - If no benchmark exists for your change, create one in `scripts/` and document what it measures.
- **Local `main` tracks `origin/main` — never let it drift.** Land on main via a PR (canonical) or commit on a lane branch and `git push origin HEAD:main`. Never build a stack of local-only commits on main (pre-commit/pre-push block **>5 ahead** via `scripts/check-main-sync.sh`; `SLO_MAIN_SYNC_SKIP=1` only when you mean it), and never `git reset` main onto a lane tip. Evidence: the 2026-10-27/28 fork — 149 local-only commits vs 35 pushed the other way = 8-file conflicts and a Next-era tree that would have reverted the Next removal if pushed.

## File Safety

- **NEVER delete user data files (notes, configs, data stores) without explicit user approval first.** Ask before removing any file that contains user-created content. Propose the change, explain consequences, and wait for confirmation.
- When consolidating or migrating data, always preserve the original as a backup before modifying.
- If a file is not git-tracked, treat it as irreplaceable — ask twice.
- **NEVER bulk-delete `~/.cache` directories by name pattern (`wt-*`, `slog-*`, …).** Many are live git worktrees — `git worktree list` is the source of truth for liveness. A heuristic glob over `~/.cache` once deleted 75 dirs including ~60 session worktrees (incident 2026-10-21). Disk-pressure recovery = package caches (`conda clean`, npm, pip), never sibling worktrees. Prevention + prune/recreate runbook: `docs/WORKTREE_RECOVERY.md`.

## Architecture

- **Notes** (`~/.config/dev-notes/*.md`) are the user's journal — source of truth for task metadata (sprint, gh, status, body).
- **Board** (`.kanban/board.jsonl`) is the kanban view derived from notes via sync.
- Sync is bidirectional: note status ↔ card column.
- **Agent sync** (`docs/AGENT_SYNC.md`) — read it first: what has landed across sessions, what is in flight, how to push your own changes; update it when you push. Local `main` must equal `origin/main` (enforced by `scripts/check-main-sync.sh`).
- **Product docs** (`docs/PRODUCT_ENGINEERING.md`) are the source of truth for what to build. Reference before creating new features, routers, or pages. User flows in `docs/UX_FLOWS.md`, persona in `docs/USER_PERSONA.md`.

## Execution Philosophy (sync core, async seam)

**The host is async; the core is sync; the seam between them is explicit.**

```
uvicorn event loop (async)
   └─ boundary: await asyncio.to_thread(...)  /  Pool.submit(...)
        └─ Engine + Pool (sync)               ← blocking work lives here
             └─ ThreadPoolExecutor / fork     ← real parallelism
```

- **`Engine`, `Pool`, `TaskQueue` are synchronous by design.** They block, and
  that is correct: the work they run is synchronous regardless, so the
  parallelism comes from `ThreadPoolExecutor`/`fork`, not from `await`.
- **The seam is async and always explicit.** One-off blocking calls go through
  `await asyncio.to_thread(...)`; sustained background work is handed to a
  `Pool` (see the comment in `apps/api/server/infrastructure/startup.py` —
  "sync hooks run in its ThreadPoolExecutor so they never starve the uvicorn
  event loop"). Sync work never runs inline on the loop.
- **Never make the core async to fix a starvation bug.** Starvation is a
  boundary defect. `async def` on the engine would wrap the same thread in
  `await` and drag the event loop into a library that must stay runnable from
  scripts, CLI, and tests with no loop. Fix the seam, not the core.
- **PGQ must stay event-loop-free.** No `asyncio` import in `pugqeep/` — it has
  to run standalone, and the async host is a _consumer_ of it, not a dependency.

### Build from scratch, import last

Execution, process management, and task management exist so we own these
primitives rather than importing them. `multiprocessing` is already gone from
PGQ (own `os.fork()` wrapper + framed socketpair channel). The one remaining
stdlib collaborator in that stack is `concurrent.futures.ThreadPoolExecutor`
for Pool workers. Before adding an external dependency to this layer, ask
whether PGQ should own it instead — and record the answer in the kanban card.

## Core Infrastructure Sync Rule

**http-client.ts is the single source of truth for all API communication.**

When backend endpoints, response envelopes, or error formats change:

1. **Update http-client.ts first** — match the new backend contract (endpoints, response shape, error codes).
2. **Propagate to consumers** — update hooks, controllers, and components that use http-client.
3. **Never bypass http-client with raw `fetch()`** — all API calls must go through `apiGet`, `apiPost`, `apiPut`, `apiDelete`, `apiPatch`, or the `request()` function.

This ensures consistent error handling, retries, caching, circuit breaking, and interceptors across the entire frontend. Bypassing http-client breaks these guarantees and creates silent bugs.

## Endpoint & Transport Rule (core-first, descriptor-projected)

**Endpoints are projections of contracts — not hand-built conveyor belts for the processes under them.**

1. **Boundary test before any endpoint** — same process → function call; different process → transport crossing. Naming the boundary is part of SOP step 4 (expert review); a new endpoint is discussed (kanban card) before code. No endpoint-per-feature by default — that is the belt we are retiring.
2. **Declare the contract once, per module** — a `ToolSpec`-shaped descriptor `{name, params (JSON Schema), result, auth_scope, idempotent, version}` is the single source of truth. `ToolSpec` (`domain/agents/_internal/tools.py`) now carries exactly this: `parameters` is stored, `params` derives the JSON Schema from it (never stored twice), and `result`/`auth_scope`/`idempotent`/`version` are defaulted fields. HTTP routes, SSE frames, CLI subcommands, agent tool-calls, and typed TS client helpers are _projections_ of that descriptor. **`create_router(spec, route, handler)` emits the HTTP half** (`apps/api/server/infrastructure/contract.py`): verb (idempotent → GET, action → POST), auth, request validation, envelope and OpenAPI metadata are all derived from the descriptor, and boot registration reads the generated `routers/_manifest.py` (`scripts/gen_router_manifest.py --check` in CI) instead of a central list. `scripts/gen_ts_contracts.py` emits the typed TS half into `apps/web/lib/protocol/contracts.gen.ts`, `packages/sdk-ts/typescript-sdk/src/contracts.gen.ts` and `apps/mobile/src/services/contracts.gen.ts` — one self-contained, byte-identical artifact per frontend consumer (freshness gated by `tests/contract/test_ts_contracts.py`), the `ts-rs` idea — a contract change that isn't regenerated fails the build instead of the user. Registering a capability = one registry entry, never an endpoint project.
3. **No god-endpoints either** — reject both N hand-written feature routers AND a single generic `POST /tools/{name}` switchyard; projected routes keep HTTP semantics, per-route auth scopes, and route-level observability. Modules self-register at boot (microkernel/plugin registration) — no central switchyard list of features in the app.
4. **Two tool tiers over one contract** — (a) **internal/model-facing tooling**: cognition/agent function-calling, `tools=[...]` model calls, the `domain/tools` everyday-tools engine (it renders model prompts — it is agent tooling, not site tooling), and the inference UI's tool list — the model sees and offers these; (b) **external utility tooling**: general site helpers and system utilities (e.g. Site Doctor: system checks + site-error fixes for dev/site-manager workflows) — tooling of the app itself, registered as contracts in the execution layer and callable by CLI, health/event stream, and operator surfaces, **never** entering the chat/agent tool registry, the `domain/tools` engine, the inference UI, or any logic the model reasons over. One descriptor contract, two views; the tiers never cross.
5. **Existing routers are grandfathered** — no retroactive purge; migrate only when a router is next touched.

## Page Rule (UI surfaces are journey-projected)

**A page is a projection of a user journey — not a trophy case of engines.**

The endpoint rule governs the API half of surface area; this governs the UI half, for the same reason — surface area is recurring cost, and "the engine exists" is not a journey. The endpoint rule has a descriptor and a parity gate; this rule historically had neither, which is how **110 `page.tsx`** accumulated against a documented target of ~20 (measured on main 2026-10-17: 19 of 20 nav routes map to a page, **91 pages are nav-orphans**, **2 of 110 carry any role check** — `PRODUCT_ENGINEERING.md` itself records "current 109 pages" against "~20 pages / ~15 routers" target).

1. **The 5-point test** — a new `page.tsx` earns existence only if ALL hold:
   1. **Named journey** — it serves a flow cited in `docs/UX_FLOWS.md` (a persona task, end-to-end). Engine existence, a new contract, or "we have an endpoint for it" are not journeys.
   2. **Not a view** — it cannot be a tab/panel/modal of a page already in that journey. Compose first; a route must justify its own journey.
   3. **Operator surfaces are tagged, not mainstreamed** — infra/system surfaces carry `role: 'operator'`. They are legitimate (the infra-workbench persona demands them); they are never silently rendered indistinguishable from journey pages.
   4. **Test asserts the journey, not the render** — a page whose only test is "it renders" is not yet justified; the journey-level assertion (or a cited existing one) is part of done.
   5. **Route budget** — `docs/PRODUCT_ENGINEERING.md` targets ~20 pages. Adding one consumes budget; exceeding it requires merging one first. Ratchet, not aspiration.
2. **Intake owns the call** — the ruling happens at card time (SOP steps 1, 4-5) by the product side, not at build time by whoever writes the file. A brief that says "build a page for X" without a journey citation goes back for the citation before code; Gene bounces it at decomposition.
3. **Enforcement is mechanical, not vibes** — `journeyId` on every `NavRoute` (`apps/web/lib/navigation.ts`) and the page baseline ratchet (`apps/web/lib/page-baseline.json`: every `page.tsx` on disk must be catalogued; new entries must cite a `ux/` journey; grandfathered and unmapped counts may only shrink) are enforced by `apps/web/lib/page-journey-gate.test.ts`. Policy lives here; the gate keeps sessions honest at 2am. (Persona matrix precedent: documentation alone did not hold the line.)
4. **Grandfathered, same clause as routers** — the existing 110 are cleanup cards' business (freeze, consolidation, role gating), never a retroactive purge by this rule.

## Command Rule (the shell CLI is descriptor-projected)

**A command's flag vocabulary is declared once and checked mechanically — not written three times and reconciled by hand.**

The endpoint rule governs the API half of surface area and the page rule the UI half; this governs the third — the shell command line. A command states its flags in a docstring, parses them in a hand-rolled loop, and draws them as RetroBIOS option rows: three writings of one fact. They drift, and this drift is the corruption class this repo already paid for — `train --merge` and `ln --symbolic` were parsed for months without ever being documented. Measured at the gate's introduction (2026-10-28): **127 commands registered**, 9 of them governed — the rest declare no long flags at all.

1. **One declaration.** A command's flags are declared exactly once. Today that declaration is the command's docstring; the destination is a `CommandSpec` descriptor (`{name, flags, positionals, subcommands, handler}`) from which the parser, `help <cmd>`, shell completion and the `options <command>` rows are all derived — one declaration, four projections, no reconciliation. Until a command migrates (clause 5), its docstring is the declaration and the other three must match it.
2. **Parity is mechanical, not editorial** — `packages/core-py/tests/test_command_parity.py` walks every registered command (explicit specs plus the auto-registered `LinuxCommandsMixin` set) and asserts _declared == parsed_, with option rows a subset of both. Commands with no flags on any side have nothing to compare and are never parametrized. Detection is static: the method source is read, never executed, because real parsers have side effects (`logs --clear` wipes the buffer, `train` starts jobs).
3. **Conventions are not declarations** — `-h/--help` is implied for every command and is exempt on both sides; a `startswith("--x-")` prefix declares the _family_ `--x-*` and must appear in the declaration. Only `--long` flags are the vocabulary: a bare short flag with no long form is invisible to the gate, so add the long form the next time that command is touched.
4. **Tier boundary** — a CLI flag declaration never enters the model-facing tool registry, `domain/tools`, or anything the model reasons over (the endpoint rule's two tool tiers). The shell describes itself to humans; that description is not an agent capability.
5. **Grandfather clause** — no retroactive purge of the existing methods. A command migrates to `CommandSpec` only when it is next touched, and positionals/subcommands stay hand-written until that descriptor exists (deferral decided 2026-10-28). Migration means: delete the hand-written parser, keep behaviour byte-identical, and let this gate prove it.
6. **The baseline may only shrink** — known documentation gaps live in `packages/core-py/tests/command_parity_baseline.json` as strict xfails: fix a gap and the node passes, strict-xfail turns that pass into a failure, and the entry must be deleted. Adding an entry is an explicit, reviewed edit — the same ratchet as `page-baseline.json`. Policy lives here; the gate keeps sessions honest at 2am.

## Test Rule (change-scoped execution, owned-code coverage)

**A test verifies code we own; a change runs the tests it affects; nothing is verified twice.**

The endpoint rule governs the API surface, the page rule the UI surface, the command rule the CLI surface; this governs the test surface — the same recurring-cost problem in its third clothing. Measured at the rule's introduction (2026-10-28): the shell suites alone are 4312 tests (~70s), and one commit paid for all of them. Suites that re-verify dependencies test someone else's contract: their release breaks ours (false positives) while the real failure — _our_ code at the seam — goes unowned.

1. **Owned-code only** — a test asserts on the behaviour of code we own (`domain/`, `apps/`, `packages/*`). Our adapter/wrapper around a dependency IS ours (the timeout we set, the fallback we wrote, the error we shape); the dependency's own behaviour is not — its suite owns it. Seam/contract tests with fakes stay: they test _our_ branching, not the fake.
2. **No redundancy** — two tests are redundant when same input + same assertion + different name; keep one. Overlapping tests with different failure modes are not redundant and both stay.
3. **Change-scoped locally, full suite in CI** — the local/PR loop runs only the affected closure via `scripts/run_affected.sh` (pytest-testmon dependency selection: which files each test actually read, mechanically recorded — no hand-maintained mapping). `RUN_ALL=1` forces a full run and re-records; unknown state (no `.testmondata`, fresh clone) is always a full run, never a guess. **Merge to main runs the full suite in CI — unchanged, the backstop.** A selective run never becomes the only line of defence.
4. **Seam gates always run when their surface is touched** — contract/parity gates (`test_command_parity.py` and siblings) are cheap and assert the seam itself; they ride along with any change to the surface they govern, selective or not.
5. **Trimming is a card, not a sweep** — tests that verify unowned internals or duplicate each other are removed per-suite behind green gates, tracked on the kanban (audit card `299d8d38`); no bulk deletion, no weakened assertions to make things pass.

# Agent Sync — read this first, update it when you push

**Purpose** — one place every session reads before starting: what has landed,
what is in flight, and how to sync your own work. Keep it short; update it in
the same commit that pushes your change.

**Last update**: 2026-10-04 ~05:30 — `feat/landing-7-cards` lands (7
infra-review cards + campaign `d083e732`, cherry-picks `2127abb6e..344168b4f`,
see Landed): startup finalizers + MogDB history + webhook facade + profiler
feed + test hygiene — tests/server **1942/1/0 vs main's 5-failed baseline**.
Earlier ~05:05: naming consolidated (card `43ca5222`):
**Mole is the one app** — never "doctor"/"watcher" as names; docs realigned
(TESTING/INDEX/FEATURES), legacy `/doctor` paths grandfathered until the
unified `mole` CLI. Earlier ~04:45: `feat/mole-watcher` lands (card `e0c80774`,
cherry-pick `e9fb80cec`, see Landed): Mole phase 1 — always-on, no-AI
monitoring (cadence, identity-set delta dedupe, JSONL journal, load-context).
Canonical 4-tree gate ≈ main baseline (833/839 nodeids
identical; the 6 extras pass standalone → shared-`/tmp` test-isolation defect,
card `6369c03e`). Earlier the same night: `feat/avion-transcript-batching`
landed (card `68ea8b9d`, cherry-pick `371025464`, see Landed): group-commit
transcript flush, benchmark green at baseline (quiet window, load 1.6); and
`fix/boot-overlay-stall` (card `cb089b43`, cherry-pick `8acb530be`, see
Landed) — the boot overlay can no longer hang forever; firefox ux-flows
benchmark **13/13**. Earlier 2026-10-03: `fix/journey-test-gates`
landed on user sign-off (merge `e081ea9f4` + test stabilization `175d7db8c`,
see Landed).
Earlier the same day: the **avion stack** — cards `13f50db7` journeys takeover
+ `e26dc68c` drivers + `24676ff2` event logger + `553da7a7` agent loop, see
Landed — and downcraft compression (card `19cd41dc`); journey suites repointed
to `:5173` (gate9's route-smoke F-class was a stale port, not a missing
server). Gates and benchmarks now run on conda `sloughgpt` + `PYTHONPATH` —
the root `.venv` is NOT authoritative (its green was hiding 2 zstd failures;
`zstandard` is now installed into conda, user-approved).

## Sync recipe

```bash
git fetch origin
git merge origin/main            # in your worktree — keep yours current
# ... your work on feat/<name> ...
git push -u origin feat/<name>   # push your own branch when done
```

- `main` is checked out at `~/Documents/sloughgpt-api-std` — land feature
  branches to `main` from there (it was clean and idle when last used); say so
  on the kanban card first if you take it over.
- Shared toolchain: ONE root `.venv` (worktree `.venv` entries are symlinks to
  it), ONE root `node_modules`, ONE `packages/*` + `apps/*` tree. Never rebuild.
- After landing: update this file, move your kanban card
  (`python -m app_planner …` + `sync`), and keep `docs/INDEX.md` current for
  new docs.

## Landed on main (origin/main = `344168b4f`)

- **2026-10-04 · `feat/landing-7-cards` lands — the infra-review follow-ups,
  surgical cherry-pick onto main.** Cards `223001e4` `d1f544fb` `ad9ef322`
  `b753cad5` `8349d901` `65d9e7ee` `832efdda` (+ campaign `d083e732`), 8
  commits `2127abb6e..344168b4f`: boot finalizers ⑨⑪⑫⑬ (history deadlock
  fixed; gzip at the seam — 4 MB stall 269.6 → 4.5 ms), startup history →
  MogDB (12 records migrated, JSON kept `.bak`), `startup_webhooks` →
  stateless facade over `WebhookStore` (shape-oracle pinned), profiler feed ⑩
  (per-hook timings, non-mutating `get_summary`, module-level `profile_hook`),
  test hygiene (stale patch targets at consumer read points + TestExecutor
  precondition fixture). Lineage fork `f27654e54`, so every pick was resolved
  against main's newer machinery: **main's REST contract, public module paths,
  and plain `WebhookStore.register()` win; only my `+` lines land** (one of my
  new tests adapted JSON-body → query-param POSTs). **Gates:** tests/server
  **1942 passed / 1 skipped / 0 failed vs main's 5-failed baseline** (+37 new);
  root `tests/` 13f/5e byte-identical to main's baseline (pre-existing); ruff
  clean; compression benchmark ~60×. Gotcha: `apps/api/server/tests` glob has 6
  order-dependent `tests.test_support` collection errors — pre-existing on
  main (A/B proven), off testpaths.

- **2026-10-04 · `feat/mole-watcher` lands — Mole phase 1: always-on, no-AI
  monitoring.** Card `e0c80774`, cherry-pick of `1b1ce40c6`: new
  `domain/core/_internal/mole/` — `run_watch` probes on a cadence over the
  existing `run_doctor`/`PROBES` (probe registry untouched; a newly
  registered probe is picked up next tick). Findings are fingerprinted by **identity set**
  `(source, check, severity, component)` — jittering payload counters (p95,
  health score, frame sizes) never re-alert (a content-hash re-alerted on every
  tick in the live smoke — caught before landing); every tick journaled to
  `$SLO_MOLE_JOURNAL` JSONL with `context` (loadavg + cpu_count — context,
  never a finding); events fire only on change; a failed tick is journaled and
  contained (the loop never dies mid-run). CLI
  `python -m domain.core._internal.mole` (`--interval/--max-ticks/--skip/
  --journal/--strict/--quiet/--json`); suggest-only, never applies. Docs:
  TESTING.md section + INDEX row (FEATURES deferred until it has an app
  surface). **Gates:** mole 6 + doctor 41 + router 9 green, ruff clean, CLI
  smoke (tick 1 baseline / tick 2 silent on the live stack; 4.8 ms per-tick
  overhead @1000 findings), canonical 4-tree suite **642 failed / 197 errors
  vs main baseline 838 — 833 nodeids identical; all 6 extras pass standalone**
  (shared-state flakes — root cause card `6369c03e`: `db_path.parent^3` escapes
  `tmp_path` into shared `/tmp/pytest-of-mana`, a 22-entry cross-session journal
  with a timeline-matched first write; their baseline has 5 unique flakes of
  its own).

- **2026-10-04 · `feat/avion-transcript-batching` lands — transcript writes are
  group-committed.** Card `68ea8b9d`, cherry-pick of `fb7859ced`:
  `AgentConfig.transcript_flush_steps` batches JSONL transcript writes — write
  every step, `flush()` every N (default 1000). **Benchmark (2k steps/scenario,
  quiet window load 1.6, conda)**: group-commit **20,806 steps/s (48.1 µs/step)
  ≈ bare loop (20,621 / 48.5 µs) and +34.4% over per-step flush (15,481 /
  64.6 µs)**; bare and per-step both match the recorded baseline (21.7k/15.7k →
  −5% / −1.4%); callbacks free (21,881); screenshot 19,576. Gates: avion suite
  green (1 isolated load-flake, passes alone) + ruff clean. An earlier run at
  load ~4 showed every row ~2× depressed — record loadavg with any benchmark
  number or the comparison is meaningless.

- **2026-10-04 · `fix/boot-overlay-stall` lands — "boot overlay never
  dismissed" is fixed.** Card `cb089b43`, cherry-pick of `d897399e7` (only the
  overlay commit lands — its branch is stacked on `feat/avion-transcript-batching`,
  which awaits its quiet-window benchmark, see In flight). `StartupOverlay`
  gains a progress-based stall watchdog: `STALL_TIMEOUT_MS = 20_000`, keyed on
  `stage:modelProgress` (mere health ticks don't reset it) → `overlay_timeout`
  event, 600 ms fade, unmount + key-deduped global banner (`startup-degraded`,
  warning, Retry = reload via `useBannerStore`); the ready-path fast exit and
  the 8 s stuck-UI path are unchanged. **Validation**: overlay suite 10/10 (+
  banner/GlobalBanner 16/16), tsc + eslint clean, full web suite **841 files /
  8068 tests green**, firefox ux-flows benchmark **13/13** (baseline was 5/13
  with 8× `boot overlay never dismissed`; now 42 console errors / 0 network) —
  run a worktree stack with `vite --port 3000` + `SLO_WEB_URL=http://localhost:3000`,
  see the CORS gotcha.

- **2026-10-03 · `fix/journey-test-gates` landed (user sign-off).** merge
  `e081ea9f4` (46 files) + stabilization `175d7db8c`: conda-first
  `scripts/python` resolver (`domain.shared.find_server_python`), journey
  module `skipif` server gate + `_web_is_ready`, `hf-*` import tests restored
  (a preserve-commit had regressed them to `kaggle-*`), `SMOKE_ROUTES`
  restored (+28 route smokes), stale `Import`→`Add file` labels, gateway
  `:8080`-aware API assertion, bounded poll-waits for fixed-sleep races.
  Validation (conda + `PYTHONPATH`, load avg 11.6): **journeys 103/103,
  computer_use 21/21, e2e 9/10** — sole red is `config_elements` (training
  config gated behind the guided-setup flow; UX-owner call, drift card). This
  push also carried the avion session's own main-merges (they had merged but
  not pushed).

- **2026-10-03 · cards `13f50db7` `e26dc68c` `24676ff2` `553da7a7` — the avion
  stack lands** (4 branches, carried on shared main by the gates session's
  push):
  - `feat/avion-journeys` (takeover of the dormant journeys card): the 6
    journey commits land; the `test_user_journeys.py` overlap with
    `fix/journey-test-gates` was resolved by keeping the avion port and
    porting all 5 gates deltas into it (`scripts/python` usage, `:5173`
    default + comment, `_web_is_ready` + module skip-guard, 4 training routes
    → 103 tests, DatasetsImport poll already equivalent in avion form).
  - `feat/avion-unify-drivers`: journeys / `run_ux_flows.py` /
    `screenshot_headers.py` drive avion `SyncRunner` instead of ad-hoc
    Playwright (`force=`, `PageControls`); `arken`/`voyager` shims remain.
  - `feat/avion-event-logger`: stdlib-only write-through JSONL journal +
    modular `EventSink` API — `tests/test_stdlib_only.py` proves it never
    imports `domain/`.
  - `feat/avion-agent-loop`: bounded awaits, honest deadline stop, retry only
    the pure-predict path, transcript write safety.
  - **Gates (conda + `PYTHONPATH`)**: avion **366 passed, 1 skipped** (skip =
    the no-websockets degradation test, env-conditional — correct here);
    live journeys **103/103** (avion port validated pre- and post-
    stabilization); ruff clean. **Benchmarks (authoritative env, load <3.5)**:
    agent loop 21.7k steps/s bare / 15.7k with JSONL transcript; event logger
    86.7k ev/s — broken sink isolated at ~journal-only, +5 sinks costs ~5%;
    downcraft e2e sha256 `ok` incl. the zstd-3 row.
  - **Insight → candidate**: residual journey failures that bucket to stale
    selectors are selector drift — a selector-contract probe validating
    journey selectors against the live DOM would fail loudly instead of as
    scattered journey F's.

- **2026-10-03 · card `19cd41dc` — download compression lands in the downcraft
  path.** `feat/downcraft-compression`: decoded byte-space enforced across the
  compressed resume path (Range offsets vs decompressed bytes; wire-space 206s
  no longer corrupt resumes), streamed `resp.read()` + stale-Range duplication
  fixes, SLZ4 resume verifies SHA-256; new
  `scripts/benchmark_downcraft_compression.py`. Gates on the merged tree: 482
  passed + ruff clean + benchmark green.

- **2026-10-02 · card `20260928_075` — full `packages/core-py/tests` no longer
  hangs.** One merge of `feat/infra-thread-teardown` covers
  `fix/pretrain-thread-leak` + `feat/card-hash-chain`:
  - **L1**: pretrain BLAS-thread leak → `os.fork()` deadlock (multimodal dedupe +
    `stop_pretrain` + cancel + conftest teardown).
  - **L3**: idle-manager / fire-and-forget / feedback-workflow / pugqeep stopped
    per test via autouse `_stop_infra_threads` (gated by
    `tests/test_infra_thread_teardown.py`, 9 tests).
  - **faf `shutdown()` fix**: `_STOP` now delivered when the bounded queue is
    full — previously each full-suite session leaked 2 `faf-worker` daemons.
  - **hash-chained kanban ordering**: `chain_index/chain_prev/chain_hash` +
    `verify_chain` (`packages/app-planner`, phase 2).
  - **gate**: chunked resumable runner
    (`~/.cache/slog-journeys/gate_chunks/run_gate.sh`, survives reboots) —
    **683F / 41629P / 0E** vs run7 baseline 594F / 41575P / 141E. Playwright
    error class eliminated (chromium installed into shared
    `~/.cache/ms-playwright`); per-file parity: 82 identical, 19 differing
    (3 converted error-class files + 16 improved).
- **2026-10-02 · local-main backlog published**: health-payload trim
  (10 MB → ~300 B), `:3000` → `:5173` doc claims, doctor Phase A (5 commits
  that sat unpushed).
- **2026-10-03 · journey suites repointed `:3000` → `:5173`** (gate9's
  route-smoke class = stale port; vite lives on `:5173`, card `e47e19ee`
  deferred core-py journey tests). `test_user_journeys`,
  `test_computer_use_training_integration`, `test_e2e_training_trigger` now use
  `BASE = os.environ.get("SLO_WEB_URL") or "http://localhost:5173"` (code edits
  preserved mid-run by the doctor session's `6b74a4024`; results `de2fe6cf4`).
  Live-stack census **95P/11F** (of gate9's 118: 83 green, 11 stale selectors,
  24 no longer exist — pre-merge file collected 99, main's consolidated file
  collects 75).
  Residual 11 = datasets-import dialog/Kaggle ×7, tools selectors ×2,
  Vite-proxy API URL, training config → journey session, card
  `20260929_journey_gates_conda`.
- **2026-10-02 · `fix/doctor-ui` merged** (`40ba857e0`) — /doctor API + page.
- `zzz_test.py` at repo root is a stray from origin's `89e01b58c`
  (shell lazy-loading refactor) — not ours; owner please remove.

## In flight

- Root-repo session on `feat/pipe-bounded-execution`; ~40 `feat/*` worktrees
  active — `git branch -vv` + the kanban board name the owners.
- The 683 baseline failures are known drift → card `56b49cf1` (drift baseline).
  Check it before treating a failure as yours.

## Gotchas that cost hours (add yours here)

- **`PYTHONNOUSERSITE=1` (AGENTS.md) breaks browser suites**: playwright is
  installed in `~/.local/lib/python3.12/site-packages/` (user site), not in
  the conda env — with the flag, `import playwright` dies and every
  Playwright suite fails at fixture setup. Browser suites
  (`test_user_journeys`, `test_computer_use_*`, `test_e2e_*`) currently run
  *without* the flag — green, no `tests`-shadow hit observed for
  `packages/core-py/tests`. Fix = install playwright into the conda env
  (needs user approval), then the flag can apply everywhere.
- `app_planner` editable install (`.pth`) points at the **root repo** copy —
  test a worktree's planner code with
  `PYTHONPATH=<worktree>/packages/app-planner/src`. **Worse: any planner
  command run from a branch that predates the chain work (e.g. the root
  repo before merging `main`) rewrites `board.jsonl` WITHOUT
  `chain_index/chain_prev/chain_hash` and strips the seal.** Chain-aware
  copies re-seal automatically (`sync()` ends in `compute_chains()`); after
  such a write, run a chain-aware `store.sync()` or `git checkout main --
  .kanban/board.jsonl`.
- Chunked-gate collect paths are relative to `packages/core-py` (nearest
  `pytest.ini`) — force-prefix `packages/core-py/`, or 16 colliding basenames
  poison `from conftest import build_test_app`.
- Never nudge a running pytest with SIGUSR1/SIGALRM (unregistered → death);
  diagnose with `-o faulthandler_timeout=150`, unblock exit by reaping
  futex-stuck pool children (`kill -9` children of the pytest PID).
- **Vite 8 dev ignores `define` → for a worktree web server on main's config,
  only origin `:3000` is CORS-safe.** main's `vite.config.ts` never loads
  `.env.local` and its `define` doesn't fire in the Vite 8 dev pipeline (the
  root branch works around it with `inlinePublicEnv` → edge gateway `:8080`,
  `ACAO: *`), so a worktree instance computes base `http://localhost:8000`;
  FastAPI's allowlist (`SLO_CORS_ORIGINS`, default `3000,8000`) then rejects
  `:5175` — measured: 5487 console errors, chat textarea stuck disabled,
  ux-flows collapse to 6/13. Fix: `vite --port 3000` +
  `SLO_WEB_URL=http://localhost:3000` → 13/13.
- **`next lint` no longer exists in this Next version** (parses `lint` as a
  directory) — pre-existing repo breakage; the lint gate is root
  `node_modules/.bin/eslint <changed files>` run directly.

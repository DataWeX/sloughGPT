# Agent Sync — read this first, update it when you push

**Purpose** — one place every session reads before starting: what has landed,
what is in flight, and how to sync your own work. Keep it short; update it in
the same commit that pushes your change.

**Last update**: 2026-10-08 ~00:35 — `fix/conftest-collection` **lands on
main** (push `b4ab962e1`, card `5c909109` → done): **conftest-shadowing
collection fix** — the 8 router test files in `packages/core-py/tests/`
(self_train/souls/system/health/inference/kb/mobile/models) now import
`from tests.conftest import build_test_app` instead of the bare
`from conftest import ...`. Root cause A/B-proven (same commit, same env):
under `--import-mode=importlib` a bare `conftest` resolves by a fresh
`sys.path` walk — no `sys.modules['conftest']` seed — and with the
repo-root config (`-c $PWD/pytest.ini`, or any arg set spanning the repo
root: the canonical gate shape) the repo-root conftest force-inserts
`apps/api/server/tests` at the path front, so the walk hit the server-tests
conftest (no `build_test_app`) instead of the `packages/core-py/conftest.py`
re-exporter; without `-c` the auto-discovered `packages/core-py/pytest.ini`
registers the re-exporter under the bare name and it passes. Invocation-
shape artifact, NOT a code regression (conftest/pytest config byte-identical
across the window; 3-dir vs 6-dir PYTHONPATH and worktree layout
irrelevant) — the same adjudication as slog-gates/progress.md. The 3
`build_test_app` collection errors that reproduced on unmodified main in
the Mole gate are gone at the source. Gates: RED 8/8 errors under `-c` →
GREEN 0 collection errors under `-c` / no-`-c` / multi-tree, full
`packages/core-py/tests --co` = 43073 collected / 0 errors, 28 test-level
failures byte-identical across shapes and all in `main-baseline-bad.ids`
(zero new), ruff + py_compile clean; the 3 residual 3-dir `mogdb` errors
are A/B-proven pre-existing (PYTHONPATH gap, attempt1-era class).
Earlier 2026-10-06 ~10:45: `feat/mole-rename` **lands on main**
(merges `78be4e601` + `26e5a1ee0`, card `752b27ab` → done): the full
**doctor → Mole rebrand** — `domain/core/_internal/doctor/` **merged into
`mole/`** (one package: probes + report + watch, cycle-safe imports),
identifiers `run_mole` / `MoleReport` / `MoleRouter`, URLs `/doctor/*` →
`/mole/*` (manifest + contract-baseline regenerated), web route
`(app)/mole` + `MoleSummary` / `RunMoleButton` + nav `nav.mole` in every
locale, CLI `sloughgpt system mole` (validator class `Doctor` →
`Validator`), tests renamed (`test_mole_probes.py`, `test_mole_router.py`),
docs FEATURES/TESTING/INDEX/TRANSPORT reworded. **ONE CLI entry** (user
decision): default = one-pass sweep + printed summary, `--watch` = cadence
+ journal, no subcommands. Grandfathered kept: `$SLO_DOCTOR_*`,
`~/.cache/slog-doctor/`, `scripts/test-doctor.py`, phoneme word-data,
historical records — **import renames break in-flight branches** still
using `run_doctor`/`DoctorReport`. Gates: canonical 5-tree
**green-by-baseline** — 388 bad ids vs baseline 849; all 32 new ids
attributed (18 journey route-loads + 2 whoami + 6 log-noise = environment,
A/B on origin/main identical; 3 `build_test_app` collection errors
reproduce on unmodified main; 3 flakes standalone-green), targeted 80 py +
24 cli + 31 web vitest, ruff/contract/manifest green, full web vitest
8072 passed + 3 standalone-green load flakes. Earlier 2026-10-05 ~13:15:
`feat/mole-phase2-probes` **lands on
main** (merge `74959c439`, card `1e57b2d9` → done): Mole phase 2 — two
read-only probes registered in the existing `PROBES` registry (one
registration each; watch/report pick them up automatically): **gates**
(parses newest `~/.cache/slog-gates` suite artifact, verdict vs
`SLO_GATES_DRIFT_MAX`, subset-scope guard + staleness check) and
**benchmark-regression** (newest-vs-history per kind, importing
`benchmark_results.py`'s `is_regression` — no reimplementation). No AI,
suggest-only, ambient (no web surface). Gates: canonical 5-tree run
**green-by-baseline** — 643 bad ids vs baseline 849 (PR #120 ci-baseline
fixes landed between), 4 new ids all pass standalone (shared-state flakes:
spinner/execute_code_js/ModelHealthMonitor tmp family `6369c03e`
/chat_trainer). Doctor+mole+allowlist targeted **71 passed**, ruff clean,
probe perf 0.9/0.05 ms per tick (phase-1 baseline 4.8 ms/tick). Earlier
~10:15: `feat/wip-column-rename` **lands on
main** (fast-forward `3f8465164`, card `ffaea823` → done): the kanban column
is **`wip`, never `in_progress`** — `config` maps both directions, the default
column schemas in Python **and** TS, 4 test files, package README + planner
specs, and 31 board rows re-keyed (3 `in-progress` orphans folded in) with the
card hash chain re-sealed because `chain_hash_for()` hashes `column`.
Gates: app-planner **185**, web planner **89**, tsc/ruff clean, chain `[]`.
Hard cutover — no compat alias. Earlier ~10:05: `feat/pugqeep-forkserver` **lands on
main** (merge `d996f8ebc`; cards `df5affc7`+`e69915c1` → done, notes → done):
gate run #4 green 621F ≤ baseline 683F, playwright-class E 0, 0 timeouts, no
hang; post-merge targeted re-run = identical F-set, E 0 (see Landed). Earlier ~05:30: `feat/landing-7-cards` lands (7
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

- **2026-10-04 · `feat/wip-column-rename` lands — the kanban column is `wip`,
  never `in_progress` (hard cutover).** Card `ffaea823` → done; ff `3f8465164`
  (`.wt-wip` worktree). The column name is a **primary key** joining card data,
  `config.STATUS_TO_COLUMN`/`COLUMN_TO_STATUS`, the default schemas in Python
  *and* TS (`store.py`, `kanban.py`, `apps/web` planner helpers/BuffetEngine/
  types) and every card's `chain_hash` payload — all moved in one pass; 4 test
  files, the package README and the planner specs follow. `migrate_boards.py`
  / `migrate_planner.py` keep `in_progress` only as a **legacy source key**
  mapping onto `wip` (a re-run must not resurrect it); `wip` beats `doing`
  everywhere (110 note entries use `wip`, **zero** use `doing`, and the web GUI
  already emitted `wip`). Data migrated byte-surgically — **952 card rows
  preserved**, 31 renames incl. the 3 `in-progress` orphans — then re-sealed
  with `compute_chains()` (the same call `sync()` makes by design) because
  `chain_hash_for()` hashes `column`: 29 → 0 violations, and pristine main was
  already 3-red (`[-1, 911, 912]`). Gates: app-planner **185 passed**, web
  planner **89 passed**, tsc clean, ruff clean, 0 eslint errors, `board stats`
  → `wip: 30`. **Gotchas:** (1) `.kanban/slot_history.jsonl` (root branch)
  keys slots as `column:pos` *inside* `node_hash()` — an append-only ledger,
  never rewrite its keys; (2) `board add` does **not** validate `--column`, so
  a stale session can reintroduce `in_progress`; (3) the long-lived
  `feat/create-router-projection` branch still carries `in_progress` and will
  conflict on `config.py` when it merges main.
- **2026-10-04 · `feat/pugqeep-forkserver` lands — fork→forkserver kills the
  fork-into-multithreaded class.** Cards `df5affc7` + `e69915c1` → done,
  notes → done; merge `d996f8ebc` (`.wt-l2` worktree):
  `SubprocessProcess` default `fork` → `forkserver` — the fresh child kills
  the fork-into-multithreaded-pytest deadlock class (gate run7: 60
  futex-stuck children). Module-level `_subprocess_worker` replaces the
  unpicklable `_worker` closure; pre-flight pickle check hard-fails
  (`PicklingError` before any allocation — closures/lambdas can't cross);
  `fork` stays as the explicit escape hatch; forkserver preload
  `[__main__, engine, target-module]` (steady spawn ~16 ms vs ~523 ms/child
  without — `scripts/benchmark_pugqeep_spawn.py`). Extended this session with
  card `a0ba949b`/`e69915c1`: **`VectorBE`'s bare `mp.Pool` (second fork site
  of the same class) → `mp.get_context("forkserver")` + `SLO_VECTOR_START_METHOD`
  escape** — it was the 88% gate wedge (isolated repro: parent `do_wait`,
  futex-dead workers; benchmark: one-time ~1.4 s cold start, steady state at
  parity — `scripts/benchmark_vector_backend_pool.py`); plus the
  `test_linux_cmds.py` `host`-fixture cwd leak fixed (bare `os.chdir` →
  `monkeypatch.chdir`; both gate runs sat in `test_time_no_args0` until exit —
  hygiene test `TestHostFixtureCwdHygiene` guards it) plus the same class in
  `test_shell_repl_more.py`: 3 bare `repl._cmd_cd("" / "~")` tests leaked
  `$HOME` from idx ~27600 (run #3 saw pytest cwd=`/home/mana` live) →
  `monkeypatch.chdir` + file-end `TestCwdHygiene`, and the shm
  `resource_tracker` warning spam carded as `ccad389e` (pre-existing, both
  start methods). **Gate run #3** (05:23→06:40, 77 min, no hang):
  597F/175E vs baseline 683F/0E — vector/linux_cmds/timeout classes all
  **0**; every one of the 175 E is the journeys family from
  `ModuleNotFoundError: playwright` (PYTHONNOUSERSITE hides user-site
  playwright; baseline chunks ran venv without -s) → run #4 carries
  `/tmp/opencode/pyshim` (playwright+greenlet+pyee symlinks on PYTHONPATH,
  no install). **Gate run #4** (07:05→08:34, 89 min, `gate exit=1`, no
  hang, **0 timeout blocks**): **621F/41902P/2E vs baseline 683F/0E** —
  vector/timeout/linux_cmds classes **0**; playwright-class **E 0**
  (`test_user_journeys` 103/103, `test_computer_use` 21/21; node-id diff
  removed 141 of run #3's E node-for-node). The 2 residual E = a
  pre-existing missing `route` fixture (identical ERRORs in run #3;
  `ROUTES`/`ALL_ROUTES` constants orphaned, zero `parametrize` in the
  file) → fixed post-gate with `@pytest.mark.parametrize`, E=0 verified
  by targeted re-run. The new `TestCwdHygiene` guard caught a 4th cd
  leaker (`test_pwd_after_cd` bare cd into `tmp_path`, leaving cwd in an
  empty dir → `test_tui_repl::test_dot` found no cwd dotfiles; run #3's
  cd-`$HOME` leak had masked it) → `monkeypatch.chdir` pin, both GREEN
  targeted; `test_phoneme_cli` 8F also disappeared (they had run against
  the leaked tmp cwd). `comprehensive_training_journeys` F = the deferred
  `:3000` hardcode class (card `e47e19ee` — `BASE = localhost:3000`,
  server died mid-file in run #4 after its first 9 tests passed; fully
  down post-gate, so its 19 parametrized routes join the same class) →
  post-gate tree ≤640F, still under baseline. Caution: the live root
  journal was externally reverted to HEAD mid-run (08:27, a session on
  `feat/create-router-projection` is still writing it) — run #3/#4
  evidence therefore lives in the WORKTREE journal only; `planner sync`
  is also non-idempotent (flips 2 foreign cards per call — do not run).
  Latency benchmark +414.9% vs baseline = contention artifact
  (load 19.6, 71 foreign pytest procs; live `:8000` serves root code, not
  this branch) — re-run when quiet. **Reconcile before
  merging**: this edits main's multiprocessing engine
  (`domain/infrastructure/_internal/pugqeep/engine.py`), while the root-repo
  session — now on `fix/startup-finalizers` at `ae237996e`, **225 commits
  ahead of main** — has already replaced that whole paradigm: `18c720be5`
  (owned `os.fork()` + framed socketpair channel, multiprocessing gone from
  their PGQ) + `d2454c84d` (single Pipe admission door) + `c4ecb6ff8`, all
  on `packages/core-py/domains/infrastructure/pugqeep/engine.py`. Root
  `AGENTS.md` on that branch now states the rewrite as fact ("Build from
  scratch, import last"). Different paths ⇒ textual merge is clean, but the
  two designs collide semantically on `SubprocessProcess.start`. This branch landed
  2026-10-04 with user sign-off — `SubprocessProcess.start` on main is the
  survivor; reconcile when their branch arrives. Benchmark names do NOT collide
  (theirs `scripts/benchmark_pipe.py`).

- **2026-10-04 · `fix/meta-weights-request-params` lands — requested sampling
  params are honoured again.** Card `d2dda007` → done, commit `806073e15`
  (onto `a189d78eb`). Contract chosen by the user: **feedback nudges only the
  parameters the caller left at their default; an explicitly-set parameter
  passes through verbatim.** Before this, `_apply_meta_weights` returned
  `get_adjustment()`'s absolute values, and `get_adjustment()` never receives
  the request — so with an empty feedback DB *every* request answered with
  0.7/0.85/40/1.15 whatever was asked, while `GenerateRequest` still validated
  the field (`ge=0.0, le=2.0`) and telemetry logged the request value the
  provider never received. Changes: `MetaWeightManager.neutral_weights`
  exposes the baseline so callers derive a *delta* (empty store ⇒ exactly zero
  change rather than a silent default swap); `_apply_meta_weights` gains
  `explicit` (pydantic `model_fields_set`; the WebSocket derives it from keys
  present in the frame) — **5 call sites, not the 4 the card claimed** (chat
  `:2971` was the fifth); the 5s cache now stores the **nudge**, never the
  merged result, because the nudge depends only on message+user while the merge
  depends on that request's explicit set (caching merged output would leak one
  request's explicit set into another's); the 4 `capture()` sites record
  `gen_params["temperature"]`. Tests: dropped the `_apply_meta_weights`
  passthrough patch (real code now passes explicit params through — strictly
  stronger), added nudge-applies-at-default and telemetry-records-sent-value,
  plus an autouse nudge-cache clear. **Benchmark** `scripts/benchmark_meta_weights.py`
  (new): cache HIT p50 1.64µs vs cache MISS p50 39.15µs (23.9×) — the merge
  cannot surface end-to-end. Gates: `tests/` 3156 passed / 0 failed,
  `apps/api/server/tests` 1223 / 0. **Found while verifying, proved NOT mine on
  pristine `a189d78eb` → card `37325860`:** the hygiene ratchet's
  `from __future__ import annotations` stringifies dataclass annotations, so
  `assert f.type is float` can never pass (3), plus 5 stale meta-weights
  *router* tests where `docs/routers.md:594-596` sides with the impl.

- **2026-10-04 · `fix/main-gate-green` lands — main's default test gate goes
  green (7 root-cause buckets).** Card `cbd2ffc3`, commit `ddd3b3d49` → landed
  as `70b162449` (rebased onto `81fd4d554`, moodboard-only delta, no overlap).
  Baseline on `c92adc47b`: **11 failed + 5 errors (16 red) in `tests/` → 3147
  passed / 13 skipped / 0 failed / 0 errors, three consecutive runs**;
  `apps/api/server/tests` **6 collection errors → 1223 passed** (the directory
  was dormant, off-`testpaths`, and never ran anywhere). Root causes, not
  counts: the contract gate was red because `routers/doctor.py` reached into
  `domain.core._internal.doctor{,.report}` (retargeted to the public facade;
  `default_report_path` joined the lazy map + `__all__`); `test_inference_generate`
  ×6 patched `domain.models._internal.provider.get_provider` while the router
  binds `get_provider` from `domain.models` **at import** — an inert mock — and
  the bare test app registered no exception handlers, so `raise_error()`
  propagated instead of returning 503; `test_cli_chat` ×5 imported `CLILogger`
  from `domain.logging._internal`, which never re-exports it; `test_feedback_domain`
  ×2 hit `_compute_gradients`' `engine is None → {}` guard (now driven by a stub
  engine, with a genuine sign invariant replacing a tautological assertion);
  `test_rag` ×1 was a **real product bug** — fusion applied `dense_weight=0.7`
  to a channel `ProductionRAG` disables by default, capping `combined_score` at
  0.3 so `HallucinationDetector.detect`'s `min_score=0.5` gate could never pass
  (grounding was structurally impossible, confidence always 0) → weights now
  renormalised over active channels; the `apps/api/server/tests` ×6 was a
  `tests` package-name collision → canonical `apps.api.server.tests` path
  (already used by `tests/server/test_server_api.py`). **Benchmarks:**
  `benchmark_bm25` A/B vs `c92adc47b` — recall@k 0.9833, MRR 1.0, reranked MRR
  1.0 **byte-identical**, latency within noise. Follow-ups filed: `b83a5788`
  (online LoRA is a silent no-op — the engine is never attached, yet stats
  count phantom updates) and `d2dda007` (meta-weights **replace** request
  sampling params — `get_adjustment` never receives them, so it cannot blend).
  **Gotcha:** the root repo's `app_planner` `.pth` has no `compute_chains`, so
  board writes made from the root copy leave that branch's board unchained
  (pre-existing there — base `f27654e54` predates the chain work); run board
  commands with `PYTHONPATH=<worktree>/packages/app-planner/src` from a
  chain-aware copy, and never issue two `board add` calls concurrently — the
  second silently clobbers the first.

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

- **2026-10-04 · startup overlay becomes an ambient boot layer — landed on main
  (`2fb066dce`, card `159300be`) and deployed in `apps/web/dist-vite`.** The
  fullscreen overlay is now `bg-[#0a0a0a]/80` + `backdrop-blur-sm` +
  `pointer-events-none` (its own controls — Retry, Show timing — opt back in
  with `pointer-events-auto`): the shell stays visible **and clickable** for the
  whole boot. The stuck (8 s) and stall (20 s) watchdog screens and the
  key-deduped banner were NOT demoted — they live in the overlay/banner exactly
  as before; only the layer's modality changed. Verified live: dead-core
  walkthrough on `:8082` (t=0.46 s hit-test passes through the overlay to
  `sl-app-content`; t=10.5 s "Still connecting" + Retry `pointer-events:auto`;
  t=21.5 s overlay gone + `GlobalBanner` "Backend not responding") and healthy
  `:8080` (overlay dismissed via ready path, `origins=[localhost:8080]`,
  0 failed requests, 0 console errors). Gates: `StartupOverlay.test.tsx` 11/11
  (incl. pass-through contract), full web suite 8075/8075 @ 841 files, tsc +
  eslint clean. **Deploy = `npm run build:vite` from a checkout ≥ this commit,
  copy `dist-vite/` to the repo root; `ServeDir` reads from disk, no gateway
  restart needed.** From `.wt-orb`/`fix/startup-overlay-busy`.

- **Landed on main (`d949eb509`, card `d484f48a`) and deployed on :8080** — the
  gateway serves the repo root's `apps/web/dist-vite` as its document root;
  rebuild with `npm run build:vite`, then restart `slough-gateway`. From
  `.wt-static`/`feat/static-hosting`. Detail:
  `apps/web/dist-vite` as its **document root** — file hit → asset, browser
  navigation (`Accept: text/html`) → `index.html` (SPA), data request →
  byte-relay; `/docs`, `/redoc`, `/openapi.json` stay proxied (path contract).
  Routing is by request *kind*, not path prefix, because SPA and API share the
  same top-level names. `npm run build:vite` builds **same-origin** (empty
  `NEXT_PUBLIC_API_URL`; `??` not `||` in `lib/config.ts` — a `||` fallback
  would silently restore a second origin). `main.py --web` retired: no Node is
  spawned. **Still open:** the 11 `app/api/**` handlers (9 planner, 1 calendar,
  1 nextauth) have no host in a static build — classification on the card
  before anything is ported or deleted.
- Root-repo session on `fix/startup-finalizers` (supersedes
  `feat/pipe-bounded-execution`); ~40 `feat/*` worktrees active —
  `git branch -vv` + the kanban board name the owners.
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
- **`note update`/`note new` auto-run `_auto_sync` → `store.sync()`** — one
  note edit triggers full notes↔board reconcile **plus** the chain reseal, so
  expect a WHOLE-file `board.jsonl` diff (canonical re-order + `chain_hash`
  cascade on dozens of cards) and cards moving to match note statuses. It is
  not a clobber — diff per-card `column` fields before reacting. Two traps:
  sync is non-idempotent (it flips cards `1f24a4f7`/`243016c3` every run —
  revert unintended foreign flips before committing: `git checkout` the
  board, then `move_card(<full-uuid>, <col>)` + `compute_chains()`), and
  `move_card` matches the id **exactly** (an 8-hex prefix returns `False`
  silently; resolve the full uuid first).

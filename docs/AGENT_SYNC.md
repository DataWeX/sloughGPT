# Agent Sync — read this first, update it when you push

**Purpose** — one place every session reads before starting: what has landed,
what is in flight, and how to sync your own work. Keep it short; update it in
the same commit that pushes your change.

**2026-10-11 ~06:30 — PROVIDER ROUTER CHAT DRIFT FIXED (branch
`fix/provider-processor-flakes`, card 9e0483b3 item 4).** The lone remaining
failure the cleanup scan flagged as "needs owner": `test_chat_delegates_to_stream`
was RED because `ProviderRouter.chat` had drifted to `await text_provider.chat(...)`
(a duplicated processor loop with **no tool rounds**), while the Jul-25 test pins
the original contract — chat is a blocking adapter over `chat_stream`. Restored
the consumer form (all providers implement both protocol methods; prod has zero
router.chat callers — the HTTP layer uses `get_provider()` raw). Gates: target
1/1, test_provider_processors 49/49, Router-touching suites 567/567, adjacent
raw-provider suites 78/78, ruff check clean (file's `ruff format` dirt is
pre-existing at HEAD). slonet-path test already green at HEAD.

**Last update**: 2026-10-09 — card `ff9ef505` → review: **generic embedding
provider config restored on main** — `api_key`/`base_url` + the
`openai_api_key` alias sync on `EmbeddingConfig`, `OpenAIEmbedder`
`base_url` wiring, and `tests/test_embedding_config.py` (8 tests).
Root cause: the feature landed on lanes as `bab6c42ed` (card `90839119`)
but main never received it, and `81616a6f7`'s sweep removed the same
fields from main — **41 local branches carry the tests but not the
fields (6 red each, incl. the root full-suite checkout)**.
Cherry-picked `bab6c42ed` onto main. Green: embedding 8, config 34,
consumer surface (vector_store/infrastructure/rag) 61+1 skip,
vector_router 37, ruff clean. Red branches heal at their next merge
from main.

**Last update**: 2026-10-07 — card `d994d4f8` → done: `GET
/knowledge/rag/documents` now pages (**default 200, `?limit`/`?offset`,
`?all=1` escape hatch**; envelope gains `total`/`limit`/`offset`;
`RAGService.list_documents` slices *before* building per-doc dicts) —
first page **31 KB vs 3,080,544 B encoded (99×)** measured on the live
19,703-doc index; the knowledge page Sources panel shows "Showing X of
Y" + Load more instead of downloading/rendering everything. Contract
baseline regenerated (+3 query params, +422; 54/556/492 unchanged).
Tests: 233 Python + 29 vitest + tsc/ruff green. Live: cherry-picked onto the root checkout's branch (now
`feat/dev-ai-unified`) as `57098d898` — 7 files; baseline + paging
tests stay on main — active from the next server restart.

**Last update**: 2026-10-06 ~10:30 — card `625d85e5` → done, **live gap
closed** (fix cherry-picked onto the root checkout as `98220d30f`;
live `/health` 10,099,639 B → 1624 B)
(investigation only, no new code; primary fix was already on main as
`2c460cbaf` since 10-01): systemic ≥1s slow-log tail **root-caused** —
`GET /health` shipped a ~10 MB `quantization.per_tensor` blob that
FastAPI JSON-encodes **on the single event loop** (2,106 calls/123 min
≈ 179 MB/min from 3–4 persistent pollers), stalling every in-flight
request: 61% of the 30s `SLO_DUMP_STACKS` stack dumps landing inside
stall windows caught the main thread mid-encode (baseline 21%), 25
same-second multi-endpoint slow clusters, 188/207 warnings were small
collateral victims. Amplifier: host CPU starvation from parallel
session test runs (PSI cpu peaks 75–80%); secondary: HF-hub
`getaddrinfo`/`list_hf_models` on request paths (`/datasets` 1.83s for
36 KB). **LIVE GAP CLOSED 10-06 ~10:08** (user-approved): the fix's
health files were cherry-picked onto the root checkout's
`feat/soul-path-grammar` as `98220d30f` (patch applied clean despite
that branch's Field-registry divergence; 49/49 + 3/3 tests on-tree;
radar-polygon half left for the regular merge) — live `/health` after
restart = **1624 B** with `model_loaded=True`, no `per_tensor`.
Same window also produced COORD card `c550503c` (live :8000 wedge
cycle evidence: accept queue 0→191, faulthandler dumps starved
mid-write = GIL-held block; OOM/journald ruled out). Follow-ups
filed: `d994d4f8` (rag/documents 2.8 MB paging), `14d26c75` (HF-DNS
cache/fast-fail), `8db4031b` (slow-log carries PSI/loadavg). Method +
evidence in the card (journalctl + faulthandler dumps; py-spy blocked
by ptrace_scope=1).

Earlier ~03:05: `feat/board-read-column-guard`
**lands on main** (fast-forward, card `b538ecbd` → done): `load_board()`
now validates every card against the board's **own schema-header**
vocabulary — an undeclared column (hand-typed; the file is editable by
design) logs a warning and re-homes under `todo`, instead of letting
renderers silently drop the card (`board show` iterates header columns
only; stats still counted it → the two views disagreed). Never raises:
one typo must not break the board. Completes the split fix: write-path
`validate_column()` = `67c406ee2` (card `46f0fac4`, other session), data
repair of the 3 `in-progress` orphans = `86fb9a300` (same session);
read-path guard was the missing half. Note the branch vocab divergence
(main header `todo/wip/review/done`, root header `.../in_progress/...`)
— validating against each board's own header makes the guard
branch-agnostic; both live boards are internally consistent today (0
coercions). Verified: 5 new tests (`test_board_column_guard.py`,
incl. a repo-board canary) + full app-planner suite **205/205**.

Earlier ~02:10: `feat/stats-local-day` **lands on
main** (fast-forward, card `7e708a68` → done): the stats "today" count
no longer reports 0 during the first hour after local midnight.
`Note.date_str` sliced the **UTC**-stamped `created_at` while
stats/filters compare against `date.today()` (**local**) — during
00:00–01:00 WAT the two dates disagree (a midnight-only flake,
invisible to daytime CI; `test_stats_endpoint` failed only in that
window). Fixed at the single derivation point: aware stamps now convert
to the local calendar day (`astimezone().date()`), naive/malformed
stamps keep the historical slice, so all four comparison sites plus the
export/grouping displays inherit local-day semantics in one place. Ids
and stored data untouched; kanban `card.due_date` (a different date
source) deliberately not touched. Evidence: live before/after with
`TZ=America/Los_Angeles` forcing the mismatch — endpoint `assert 0 == 2`
FAIL → PASS; full app-planner suite **200/200 in default TZ and TZ=LA**;
6 new deterministic `TZ`-forced unit tests
(`tests/test_note_date_str.py`, both conversion directions + fallbacks).

Earlier ~01:25: `feat/kernels-nogil` **lands on
main** (fast-forward, card `f30be275` → done): all 16 builder kernels in
`domain/training/_internal/slonet_kernels.py` are now
`@njit(cache=True, nogil=True)` — a thread-domain call to any hot kernel
**releases the GIL** instead of starving the host loop for the whole run
(the demonstrated outage mechanism, now measured at real scale: a stock
prompt-scale `fused_attention_multi` call held the GIL for all ~42 s of
its duration). New gate: `scripts/probe_gil_hold.py` (the card's
referenced `benchmark_isolated_execution.py` never landed) answers the
decision question directly — standalone twins: control thread stalled
**100%** without nogil vs **0.5%** with it; `--real` mode probes the
production decorator: **0.4%** of a 2.6 s call. Numerics are
**bit-identical** (6/6 kernels, `max_abs=0` stock vs nogil);
`benchmark_slonet.py` **7/7** (`logit_diff 0.0000`, `max_output_diff
0.0000`); kernel suites 122 passed in both env protocols with the 2
failures proven pre-existing/by-design (`test_causal_mask_effect` fails
on the stock file under numba-hidden; `test_check_numba_false_branch`
asserts numba-absent). First run recompiles (cache-signature change, one
time, expected). Findings for follow-up: (a) numba is **user-site only** —
the repo-standard `PYTHONNOUSERSITE=1` protocol silently runs the numpy
fallback and never exercises the compiled kernels; (b) the
`benchmark_latency.py` gate could not run — the shared `:8000` `/chat`
hangs while `/health` reports healthy (reproduces after a server restart;
PGQ-wedge territory, another session's lane) and the host sat at 97°C
thermal clamp; (c) pre-existing perf oddity: prompt-scale
`fused_attention_multi` (512×512×8×64) takes ~40 s/call **stock and
nogil alike** — the docstring's einsum note understates it at that shape.

Earlier ~12:55: `fix/vector-shm-tracker` **lands on
main** (fast-forward, card `ccad389e` → done): VectorBE's `/psm_*`
resource-tracker warning spam is fixed at the **topology** level —
`VectorBE.__init__` now calls `resource_tracker.getfd()` before starting the
pool, giving the fork arm the same shared-daemon wiring forkserver
(`forkserver.py:92`) and spawn (`spawn.py:119`) children already get.
Root cause: `SharedMemory` registers *attaches* as well as creates
(gh-82344); fork workers forked before the parent first touched the tracker
inherited `_fd=None` → a private tracker daemon per worker → exit cleanup
re-unlinked every attached segment (118 warning lines from one
matmul+softmax+rmsnorm probe). Sharing the creator's daemon makes attach
REGISTERs idempotent against the creator's claims, so the create/unlink
pairing alone governs cleanup. An earlier attach-site `unregister()` fix was
proven *wrong* by topology evidence — in the shared topology it clobbers the
creator's claims (8 KeyErrors, one per segment). Verified: **0
`resource_tracker` lines in fork / forkserver / spawn arms**, no `/dev/shm`
leaks, `test_vector_backend.py` **24/24 in default and fork-env runs**; the
suite's 2 env-hatch blind spots fixed (defaults test + `vector_be` fixture
now `delenv` `SLO_VECTOR_START_METHOD` — the fixture pin also guards the
documented live-fork-in-suite deadlock). Follow-up on the card:
`scripts/benchmark_vector_backend_pool.py` (the finder-session script the
card references) still does not exist.

Earlier ~08:15: `fix/journey-server-skip` **lands on
main** (fast-forward, card `b9f12ad2` → done): the serverless gate block in
`packages/core-py/tests/test_user_journeys.py` +
`test_computer_use_training_integration.py` is now **124 skipped in 12s,
EXIT=0** (was ~113 errors plus a 90s × 2 dead wait) — both
`ensure_servers_ready()` fixtures now do the repo-blessed capability gate (5s
probe → `pytest.skip("API server not running")`, the `test_integration.py` /
card `6e826226` convention) instead of asserting live servers. Chromium itself
was already healthy (installed 10-02; live launch `151.0.7922.34` — the card's
original "missing executable" premise was stale). SECOND defect found while
verifying: `test_user_journeys.py` was **uncollectable under
`--import-mode=importlib`** (root-ini addopts) — importlib's rootdir-relative
`tests.*` module name resolves through cwd's repo-root `tests/` **package**
(it has `__init__.py` + its own conftest), so `packages/core-py/tests/conftest.py`
never executes its avion `sys.path` insert → `ModuleNotFoundError: avion` at
collection (probe evidence: avion never in sys.path under importlib, present
from sessionstart under plain mode). Fixed file-locally with the same
source-bootstrap idiom both conftests already use (no installs, mode/cwd
independent); the underlying duplicate-`tests/` naming collision remains card
`d2803a3c`'s scope. Verified: both modes 124 skipped EXIT=0; ruff +
py_compile clean.

Earlier ~03:27: `fix/root-tests-order-dependence` **lands
on main** (fast-forward `0246b027c`, card `5473eb90` → done): the rotating
order-dependent `tests/` failure is **root-caused — it was never a
module-level global; it was a path-depth off-by-one**. `tests/test_api_keys.py`
computed `_server_dir` with `parents[2]` (the idiom for `tests/server/`-depth
files), which at `tests/` depth overshoots to the PARENT of the repo — in a
nested worktree (the gate layout) that's another checkout, so it put the parent
repo's `apps/api/server` at `sys.path[0]` **twice** (the second insert was
unconditional). `training` then resolved cross-tree, `find_repo_root` read the
parent's `data/`, the dataset lookup missed, and `raise_error(400)` escaped the
tests' bare `FastAPI()` as an unhandled **500** — so `>=400` assertions passed
while `==200` ones failed: the rotating signature. Plain checkouts get a dead
path (dormant → "stopped reproducing"); `tests/server/` scope was accidentally
protective (imports `training` first, `sys.modules` shields the rest). Fix:
`parents[1]` + drop the duplicate insert; same class in
`packages/core-py/tests/conftest.py` (`parents[2]` → `parents[3]` — dead
everywhere, while line 23 already had `parents[3]` right for avion); both
`_make_client` builders now use conftest's `build_test_app` so `raise_error`
yields 4xx, not 500 (the `ddd3b3d49` precedent). Gates: deterministic **2-file**
repro 3 failed → 33 passed; non-server scope **1224 green** (was 1221+3); full
scope **3166 green**; plain-layout worktree 1224 green; core-py collection exit
0; ruff clean. `chat_trainer::test_generate` never reproduced after `ddd3b3d49`
(green in every full run since) — no mechanism was ever observed for it, so the
card records it as not-reproducing, not fixed.

Earlier ~19:20: `fix/core-py-feedback-suite` **lands on
main** (fast-forward `ce02af883`, card `37325860` → done): the 8 pre-existing
feedback-suite failures are fixed, and **both clusters were test-vs-impl
drift, not code bugs** — no production code changed. Cluster 1:
`meta_weights.py:8` / `model_health.py:14` carry the `from __future__ import
annotations` ratchet, so dataclasses record `f.type` as the *string*
`"float"` and `assert f.type is float` could never hold; the tests now
resolve with `typing.get_type_hints()` (repo-wide grep: no other file uses
the f.type-identity pattern). Cluster 2: `k` defaults to **5** — schema,
web controller (`k = 5`) and `meta-weights-controller.test.ts` all agree, the
test's `10` was stale — and the router registers `/meta-weights/ping|get|stats`,
exactly what `docs/routers.md` documents. Gates: the 2 files **153 passed**,
ruff clean. Two findings for other cards: (a) the 3 health failures that
surface next to them are pre-existing on pristine main *and* order-dependent
(8 when run alone → 3 with siblings) → card `5473eb90`; (b) core-py's
documented `-n auto` gate (`docs/TESTING.md:10`) cannot run here —
**pytest-xdist is not installed** (install needs user approval), and the
suite is 42,568 tests, so serial runs exceed 30 min.
Earlier ~16:45: `fix/board-column-validation` **lands on
main** (fast-forward `67c406ee2`, card `46f0fac4` → done): the column name is
validated at the **core**, not at the CLI. `PlannerStore.validate_column()` is
now the single source of truth and every write path funnels through it —
`add_card`, `move_card`, and `update_card(column=)` — so a retired or
misspelled spelling (`in_progress`, `in-progress`, `TODO`) can no longer enter
the board; that gap is exactly how stale spellings had leaked in (only
`_board_move` ever checked). `column_names()` stops at the schema header with
a substring pre-filter, so the hot path keeps its throughput: **58µs** on a
1.23 MB / 1052-card board vs 36.1 ms for a full `load_board()` = **0.1% of an
add**. CLI `_board_add`/`_board_move` catch `ValueError` for the friendly
`Invalid column: X. Valid: …` message and the duplicated check in
`_board_move` is gone. Gates: app-planner **194** (+9), ruff clean; benchmark
no regression — faster than unmodified main in 3 of 4 paired runs while the
box sat at load 10–12 (the swing is contention, not the code). Board half on
root branch `86fb9a300`: 3 hyphenated `in-progress` orphans → `in_progress`
and the card → done, 957 cards / 959 lines preserved, 0 invalid columns left;
`verify_board` still reports only the pre-existing `todo:78` break
(2026-10-04T06:36Z, identical at HEAD — not introduced here, journal is
append-only so it is reported rather than rewritten). Blast radius checked:
`PlannerStore` has no caller outside app-planner, and the GUI runs on
`KanbanStore`, which already guards its own `move_card`.
Earlier ~10:15: `feat/wip-column-rename` **lands on
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

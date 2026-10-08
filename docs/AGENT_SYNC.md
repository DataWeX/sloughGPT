# Agent Sync — read this first, update it when you push

**Purpose** — one place every session reads before starting: what has landed,
what is in flight, and how to sync your own work. Keep it short; update it in
the same commit that pushes your change.

**INFRA 2026-10-08 ~06:15 — shared `.git` corruption HEALED (read this if
you ever hit "bad object" / "object file is empty").** One external event
(3 waves: 04:43, 05:15:53, 05:18 — correct size, all-zero content ⇒
zeroing/partial write outside git; cause unknown) damaged BOTH the object
store and a worktree file.
**(A) Object store** — symptom: `git fetch` fails *"object file … is
empty"* / *"did not send all necessary objects"*; `fsck` reports missing
commit/tree/blobs. Plain fetch can't heal it: remote-tracking refs advertise
we already HAVE the tips, so negotiation skips the missing mid-history
objects. Recipe: (1) delete all zero-byte files under `.git/objects` (git
skips existing paths on fetch, so poison blocks re-download), (2) `git
fetch --refetch origin main` (full pack, ignores local HAVEs), (3)
`git fsck` to verify. Blobs/trees whose content matches the worktree can
also be rebuilt via `git hash-object -w` / `git write-tree` —
content-addressing verifies the result.
**(B) Worktree torn file + stat-cache poisoning** — symptom: a tracked
file's content silently becomes a torn write (old header + untruncated tail
remnant ⇒ duplicated paragraphs) while `git status` reports **clean**: the
crash also stamped the index stat-cache to match, so git never re-hashes
and even `git checkout HEAD -- <f>` *skips* the restore (it trusts stat).
Detect (bypasses stat): `git hash-object --path <f> <f>` vs
`git rev-parse HEAD:<f>`; repo-wide: compare `git ls-files -s` shas against
`git hash-object --stdin-paths` (hash 120000 symlinks manually — hash-object
can't open symlink-to-directory). Heal: re-write the bytes directly
(`git cat-file -p HEAD:<f> > <f>`); a plain `git checkout` will no-op while
the poisoned stat stands. Final state after heal: fsck exit 0, 0 zero-byte
objects, 5130/5130 tracked files match the index. **Do NOT rewrite refs to
"fix" corruption — forward-only recovery.**

**Last update**: 2026-10-08 ~11:05 — `5473eb90` **closed: rotating-victim root
cause was missing from main — ported.** The root-cause fix `0246b027c`
(root lane, Oct-5; its message names this card) was on 6 feature branches but
**never an ancestor of origin/main**: `tests/test_api_keys.py` used
`parents[2]` at `tests/` depth → overshoot to the PARENT checkout, inserting
its `apps/api/server` at `sys.path[0]` twice (second unconditional). Dormant
in plain checkouts (dead path), **live in nested worktrees** (`.wt-mole` →
parent checkout is a real repo). Ported via `git cherry-pick -x` →
`1eca0294a` (api_keys `parents[1]` + dedupe, core-py conftest `parents[3]`,
distill `_make_client` → `build_test_app`). **Second victim found and fixed by
the same port**: `tests/server/test_vm_router::test_returns_10_programs`
failed only in full-collection scope — `vm_builtins` resolved to the parent
checkout (imports `CAVE_GAME_ASM`, absent from this tree's `vm_programs`)
→ ImportError silently swallowed by the endpoint → `programs: []`;
predecessor-scope runs were accidentally protective (sys.modules caching),
exactly as `0246b027c` describes. Evidence: chat_trainer 19/19, distill 12/12,
api_keys 21/21, 2-file repro 33 passed, **full `pytest tests/` gate = 3226
passed / 0 failed / EXIT 0** (was 1 failed). Rule of thumb: at `tests/`
depth the repo root is `parents[1]`; `parents[2]` is OUTSIDE the repo. A
quiet-window core-py full-suite baseline run is still pending for triage
cards (`56b49cf1`, `ca167dbd`) — today's full run was abandoned (repeated
300s hanger-timeout burns under machine contention).

**Last update**: 2026-10-08 ~09:20 — `2656f858` **closed by reconciliation, not
rework** + **PYTHONPATH correction (read before judging any red)**: the card
had already been resolved 2026-10-05 on the ROOT lane's board
(feat/cave-game-vm; 6-env evidence matrix, "7/21 notifications red NOT
reproducible — keep as-is", wire test `28fe3261e`, soul_path impl) but that
board state never merged to main, so main screened it as open — claimed, then
caught at screening depth: **boards diverge per lane; screen BOTH
`.kanban/board.jsonl` copies (worktree + main checkout) before claiming.**
Ported the resolved title/description to main's card, closed with independent
re-verification: canonical path collects `packages/core-py/tests` 42494 / 0
errors and `tests/` 3226 / 0 errors; `test_notifications` 26/26 solo.
**CORRECTION to earlier entries' "13 pre-existing `[mogdb]` failures": those
were an env artifact, not baseline** — a short PYTHONPATH missing
`$PWD/packages/mogdb/src` yields 24 collection errors
(`No module named 'mogdb'`) in the two test roots AND `test_gui[mogdb]`
failures (43/43 green with the full path). Canonical gate PYTHONPATH =
`$PWD:$PWD/packages/downcraft:$PWD/packages/core-py:$PWD/apps/api/server:$PWD/packages/mogdb/src:$PWD/apps/cli/src:$PWD/packages/sdk-py`
(+`$PWD/packages/app-planner/src` for planner). Before attributing mogdb-shaped
reds to a baseline card (`ca647de2` et al.), re-run with the full path. A
background full-suite run is in flight for a fresh failure baseline
(feeding `5473eb90` rotating-victim campaign + `56b49cf1`/`ca167dbd` triage).

**Last update**: 2026-10-08 ~08:05 — `6e826226` **journey playwright capability
guard lands** (test suites only, no prod code): all 4 browser journey suites
now carry a module-level `pytest.importorskip("playwright.sync_api", …)` guard
BEFORE their driver imports, so a missing playwright = clean capability SKIP
instead of the 175 fixture-setup ERRORs of the Oct-04 sweep. Meta-test
`packages/core-py/tests/test_journey_playwright_capability.py` locks both
invariants (guard wiring/order; poisoned-playwright subprocess → skip with
reason, never error). Acceptance run under `PYTHONNOUSERSITE=1`: 192
collected, **0 collection errors**, 133 passed / 59 failed — all 59 are
`ERR_CONNECTION_REFUSED localhost:3000` (dead Next port; live web = vite
:5173, port story tracked by `257d310b`/`e157e4f1`). Note: playwright 1.62.0
is already installed in the conda env (fix 1 was done elsewhere); purging the
orphaned user-site `tests` package (fix 3) still awaits user approval.

**Last update**: 2026-10-08 ~07:05 — `fix/board-sync-idempotency`
**lands on main** (card `08baf13f` → done): **notes→board sync is now
net-idempotent.** `canonical_notes()` in `app_planner/sync.py` collapses
duplicate-title journal twins to ONE winner per title (latest `updated_at`,
ties `created_at`/`id`, then title-sorted) and feeds both `PlannerStore.sync()`
and the legacy branch the GUI `--sync` runs — reruns report `0 moved` instead
of flipping cards forever. `get_stats()` gained `uniqueIds` / `dupIdLines` /
`titleCollisions`: the hash chain stays green while id/title duplication
exists, so check these, not just `verify_chain()`. Board deduped to 946/946
unique ids (last-wins; the 3 column-conflict triples todo/wip/done kept
`done`; 3 distinct-id `titleCollisions` remain for human review). Journal
twin notes left in place (user-journal file-safety) — sync now ignores the
loser deterministically. Tests 185 → 190 passed; the 13 pre-existing
`[mogdb]` failures unchanged (baseline owned by `ca647de2`, `d2803a3c`,
`56cf354c` — not duplicated).

**Last update**: 2026-10-08 ~05:40 — `fix/kanban-phantom-column`
**lands on main** (push `91e81243b`, card `b538ecbd` → done): **read-path
column validation** — write paths already rejected retired spellings
(`67c406ee2`) but `load_board()` — every reader's choke point — accepted
anything, so a hand-edited `"column": "in-progress"` line produced a card
`board show` (header-driven) never rendered while `stats` (card-driven)
counted it: two views, two totals. Now `load_board()` warns + coerces to the
fallback (`todo`, else first declared) — coerced, never dropped; the card
becomes visible and consistently counted. Card drifted before pickup: the 3
phantom cards were already repaired to `wip` (verified, untouched) and C
follows automatically (stats reads load_board). 4 new tests incl. a
recurrence guard on the **real** `.kanban/board.jsonl` (would have caught the
original bug). Gates: RED 3 → GREEN **198 passed** (full app-planner suite),
ruff clean; DONE-WHEN proven live: show renders **966** card lines == stats
Total 966 (842/77/32/15 all in vocabulary).

Earlier 2026-10-08 ~05:00: `fix/online-lora-phantom-updates`
**lands on main** (push `93286c6fe`, card `b83a5788` → done): **phantom
online-LoRA updates retired** — the loop was dead in production (no engine
attach path, `engine=None` always) yet counted every buffer drain as a
successful update with "Updated with N samples" telemetry. DECIDE resolved to
retire: `apply_to_logits` has **zero production callers** (so even a wired
engine wouldn't reach inference — wiring is a separate feature card) and real
backprop needs engine access the interface doesn't expose. Shipped: honest
applied-vs-skipped accounting (`total_skipped_updates/skipped_samples`,
`total_updates`/`last_update_time` only move on real applies), shape-mismatched
gradients skipped instead of raising into the swallow, the pseudo-gradient
**random-noise fallback removed** (it mutated live weights), `engine_attached`
stat + not-implemented docstrings. 16 tests new/retargeted — 4 of them were
baseline reds asserting gradients-from-nothing, now green under the honest
contract. Gates: RED 8+3 → GREEN **273 passed** across 6 files
(online_lora/online_train/feedback_domain + feedback_controller/
workflow_router/quality_guard), ruff + py_compile clean, 0 frontend consumers.

Earlier 2026-10-08 ~04:20: `fix/shm-resource-tracker-warnings`
**lands on main** (push `f5ccd713a`, card `ccad389e` → done): **resource_tracker
`/psm_*` warning flood silenced** — root cause *proven* with a tracker-pid probe:
`VectorBE` forked its Pool before any shm op, so fork-arm workers each started
their **own** tracker (pid mismatch, SHARED=False) and accumulated attach
registrations (weights + unique per-dispatch out names) in per-worker caches;
parent unlinked files first → worker-exit sweeps hit ENOENT → **214 warning
lines** on the fork arm (99 `Errno 2` + 8 "leaked" notices). Fix, 2 sites with
worker functions untouched: `resource_tracker.ensure_running()` **before** Pool
construction (one shared tracker → attaches dedup against the creator's
entries) + `__del__` `FileNotFoundError` fallback unregistering `block._name`
(stdlib `unlink()` = shm_unlink *then* unregister, no try/finally) with
`_shm_blocks.clear()` idempotence (double-unregister KeyErrors the tracker —
measured). Rejected: worker-side unregister — on the shared tracker it would
pop the creator's entry. forkserver already shared via spawn prep `tracker_fd`
(0 lines before and after — the card's "both methods" didn't reproduce here).
Gates: fork arm **214 → 0 lines** (stderr 0 bytes, same args), both arms
iters=10 = 0 warnings, `/dev/shm` empty, steady matmul 29.6ms vs 37.4ms RED
(no regression); `test_vector_backend` **24 passed**; ruff + py_compile clean.

Earlier 2026-10-08 ~03:45: `feat/generic-provider-config` **lands on
main** (pushes `859b83811` + `226b35352`, card `90839119` → done): **generic
embedding provider config** — `EmbeddingConfig` gains `api_key` + `base_url`
(any OpenAI-compatible endpoint: ollama, LM Studio, vLLM) with a bidirectional
alias mirror so `openai_api_key` reads/writes keep working; env ordering
covered (`_apply_env_overrides` ends in `model_validate`, so both the legacy
`SLO_EMBEDDING__OPENAI_API_KEY` and new `SLO_EMBEDDING__API_KEY` resolve);
`OpenAIEmbedder`/`Embedder` pass `base_url` into the client; `.env.example` +
example config updated. **Reuse, not rebuild**: the work already existed as
`bab6c42ed` on `origin/feat/embedding-provider-config` (137 unmerged commits,
2026-09-29 — card notes even said "delivered: bab6c42ed"), so it was
cherry-picked (clean). This session added 3 gap tests (legacy-env back-compat,
precedence, `base_url=None` default) and fixed 5 `FakeClient` fakes in
`test_embeddings.py` the original commit missed (TypeError on `base_url`, new
vs baseline). **Disclosure**: the feature first rode into the board-reseal
commit `859b83811` still staged from the cherry-pick (message says reseal
only) — contents recorded in `226b35352`'s message; no history rewrite on
shared main. Gates: red-green proven (pre-feature `HEAD~2`: 9F/2P → 11/11);
family `test_config` + `test_embedding_config` + `test_embeddings{,2}` = **100
passed**; ruff + py_compile clean.

Earlier 2026-10-08 ~03:05: `fix/feedback-suite-drift` **lands on
main** (push `dd83baf9a`, card `37325860` → done): **feedback suite drift
fixed** — 7 red (card reported 8; the `test_dataclass_field_types` "×2"
counts once on current main) across two root causes, both test-side, no
production code touched. (1) `test_feedback_dataclasses` asserted
`dataclasses.fields().type is float`, but `meta_weights.py`/`model_health.py`
carry `from __future__ import annotations` (hygiene ratchet) so `f.type` is
the *string* `"float"` → resolve via `typing.get_type_hints()` (pattern
verified unique: 0 other sites repo-wide). (2) `test_meta_weights_router`
was stale vs impl **and** docs (both agree): `k` default is 5, and
`APIRouter(prefix="/meta-weights")` applies at registration → routes are
`/meta-weights/ping|get|stats` per `docs/routers.md`. Gates: RED 7 → GREEN
**153/153** in the two files, full feedback + meta_weights family **547
passed**, ruff + py_compile clean.

Earlier 2026-10-08 ~02:45: `fix/vite-worker-format` **lands on
main** (push `3f73ff956`, card `60652af2` → done): **production `vite build`
unblocked** — explicit `worker: { format: 'es' }` in `apps/web/vite.config.ts`.
The WebGPU worker is a module worker (`lib/soulnet-webgpu/index.ts:62`, the
only `new Worker` site) and vite ≤7 defaults `worker.format: 'iife'`, which
code-splitting builds reject. Nuance found while verifying: the shared lock
resolves **two** vites — `apps/web` → 8.3.0 (rolldown, canonical
`npm run build:vite`, was already green) and root → 7.3.6 (the path the
Oct-04 sweep hit, bare `vite build` / checkouts without
`apps/web/node_modules`); explicit `'es'` makes the build green on **both**
(exit 0, `dist-vite/` + ESM worker chunk, entry wiring intact). Also: `.wt-mole`
was the only worktree missing the standard `node_modules` bridges — repointed
`apps/web/node_modules` (empty dir; only a regenerable vitest cache, backed
up) + worktree-root symlink to the shared root copy, matching the exact
pattern all 7 other `.wt-*` worktrees use (no install, no new tree). Gates:
RED vite7 exit 1 → GREEN both vites, `tsc --noEmit` clean, vitest
`test:lib` 1517/1517. Covers the forward risk for `870ab1bad`'s CI
"Vite build" step.

Earlier 2026-10-08 ~02:12: `fix/tokens-router-nonederef` **lands on
main** (push `d50dafa15`, card `f01852a3` → done): **tokens router identity
fix** — all 6 `/tokens/*` endpoints crashed in every auth mode:
`require_auth_if_enabled` returns `None` when `SLO_AUTH_REQUIRED` is off (dev
default) → `auth_user["id"]` raised `TypeError` (caught → AppError 500), and
JWT payloads carry `sub`, so auth-**on** raised `KeyError` too. tokens.py was
the only router in the server with unconditional `auth_user["id"]` derefs.
Fix = `_resolve_user_id()` None-safe claim fallback `sub → id → username →
"anonymous"` (mirrors errors.ingest / api_keys / users) — endpoints stay
functional with auth off; no 401-when-anonymous exists anywhere to copy.
Tests 12F → **13/13** (fixture now registers `register_app_error_handler`,
overrides the auth dependency to scope accounts by `X-User-Id` — the approach
from orphaned unmerged `ee1ba337f`, rescued — resets the billing singleton
per test, unwraps the `["data"]` envelope, asserts **422** for pydantic field
violations per `exception_handlers.py` contract, not 400) + new `TestAuthDisabled`
regression test. Router-family A/B on identical env: origin/main 111 failed →
fixed 98, comm diff = **zero new ids**; ruff + py_compile clean. Note: family-
shape failures in `test_self_train`/`test_settings` are baseline path drift
(old `tests/server/` ids, card `d2803a3c`) — pass 55/55 standalone.

Earlier 2026-10-08 ~00:35: `fix/conftest-collection` **lands on
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

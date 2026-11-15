---
id: 20261004_transport_projections_descriptor_routes
title: "Descriptor-projected transport: create_router(spec), boot self-registration, contract CI"
status: done
landed: c38c4e04b + a440727fa + fa7f62eb6 (main/api-std)
date: 2026-10-04
tags: [architecture, api, transport, tech-debt]
board_id: 57b0ea27-10ec-40ae-b67d-4893f263a95f
board: 57b0ea27-10ec-40ae-b67d-4893f263a95f
hash: 855bd75adf05525adfc6630286e6848b3acdd2072ebb65687ed86227e9921e24
---

PATTERN (2026-10-04, design discussion with user): "endpoint per feature" and "one `/tools` god-endpoint" are the same conveyor-belt mistake — transport treated as something you *build*. The answer is descriptor → projection → registration.

1. **Contract declared once, per module** — a ToolSpec-shaped descriptor `{name, params (JSON Schema), result, auth_scope, idempotent, version}`. `domain/agents/_internal/tools.py` already stores `parameters` and derives `params` (one declaration, two views): the contract half exists, `create_router(spec, route)` — the transport half — does not.

2. **Projections** — adapters emit the transport from the descriptor: HTTP route fragment (real verb, status code, `success_response` envelope, `classify_and_raise` errors, `auth_scope`, OpenAPI path — composed by `include_router` so the contract itself is modular); SSE frame kinds on the health/system/training streams; CLI subcommand; agent tool-call (`ToolRegistry` already reads `ToolSpec`); typed TS helper for `http-client.ts`. "Endpoint" stops being a noun you build and becomes a projection you configure.

3. **Registration** — microkernel: modules self-register at boot, the app carries no central feature list. Today's `_router_names` in `apps/api/server/routers/__init__.py` *is* that switchyard — grandfathered, and its deferred imports exist for a real reason (90s → 8s cold start), so self-registration must preserve lazy loading. Self-enforcing by workflow, not hierarchy: a CI contract check — JSON-Schema params + auth scope + contract test, with the OpenAPI diff reviewed like code (extend `scripts/check_docs_api_parity.py`).

**Doc written:** `docs/TRANSPORT_PROJECTIONS.md` — pattern, projection status table, target workflow, guardrails, gap table; registered in `docs/INDEX.md` and cross-linked from `docs/routers.md` + the Router Playbook. Doctrine itself stays in AGENTS.md *Endpoint & Transport Rule* (no duplicate doc).

**Why not `/tools`:** a single `POST /tools/{name}` switchyard kills GET/POST semantics, per-route auth scopes, OpenAPI paths, cacheability and route-level metrics. If one-call semantics are ever wanted, JSON-RPC 2.0 (`/rpc`) is the honest standard, not an ad-hoc switchyard. One envelope / one auth story / one versioning story come from sharing the descriptor — not from sharing one URL.

**TASK (build half — this card):** (1) `create_router(spec, route)` — helper named `create_router`, not `build_router` (user decision 2026-10-04), living on the server/web side only (`apps/api/server`); `domain`/infra core stays capability-only; (2) boot-time self-registration replacing `_router_names` via a **generated** `routers/_manifest.py` with a CI diff, without losing the lazy-import cold-start win; (3) CI contract gate = descriptor check **blocking** + OpenAPI diff **report-only** until the routers.md backlog clears; (4) TS projection into `http-client.ts`; (5) pilot = **brand-new descriptor-backed router** exposing `GET /contracts` (the contract registry's own read path — named here per Endpoint & Transport Rule 1), **no migration of existing routers**. 57 existing routers grandfathered: migrate only when touched.

**ACCEPTANCE:** one descriptor entry yields a real route with correct verb, envelope and auth_scope; registering a capability touches no central list; the contract check fails a descriptor missing `auth_scope` or a params schema; OpenAPI diff surfaces as a CI artifact; grandfathered routers behave unchanged; no new doctrine doc (AGENTS rule + Router Playbook remain the only sources).

**SHARED WORKSPACE —** index updated: `docs/INDEX.md` gains `TRANSPORT_PROJECTIONS.md` in this change; shared copy reused: root toolchain only, ZERO installs; coordination named: card `656023ab` (http-client host module) and Router Playbook seam card `97260724` referenced only, neither modified; board/note writes are append-only and preserve the in-flight dirty `.kanban` / `.dev-notes` files of other sessions.

---

**PROGRESS 2026-10-04 — deliverable (1) LANDED, branch `feat/create-router-projection`.**

Built `apps/api/server/infrastructure/contract.py` — `create_router(spec, route, handler)` — plus `apps/api/server/tests/test_contract_projection.py`. **No existing route, router or `routers/__init__.py` was touched**, so the 57 grandfathered routers are untouched by construction (acceptance line: "grandfathered routers behave unchanged").

What is derived from the descriptor, all of it:
- **verb** — `resolve_method()`: `idempotent -> GET`, action -> `POST`; an explicit `method=` override wins and is **logged** so descriptor-vs-route drift stays visible.
- **422 contract violations** — a Pydantic request model is built from `spec.parameters`; no hand-written validator and **no new dependency** (`jsonschema` is not installed and was deliberately not added; FastAPI's own validation feeds the app's existing `RequestValidationError -> 422 / E_VAL_FIELD` handler in `infrastructure/exception_handlers.py:209`).
- **envelope** — every success goes through `success_response`.
- **errors** — `_dispatch` routes every handler failure through `classify_and_raise(e, source=f"router.{spec.name}")` (test asserts `AppError.source` is tagged, not just that a 500 happened).
- **auth scope** — `Depends(require_auth_if_enabled)`; test proves `SLO_AUTH_REQUIRED=true` -> 401, disabled -> 200.
- **OpenAPI `x-contract`** — carries `{name, version, auth_scope, idempotent, params (JSON Schema), result, emitted_by}` and the `requestBody` schema, so the contract is readable out of `/openapi.json`.

Two **boot-time** guards (checks that a 422 structurally cannot catch, so they run when the router is built):
1. `check_handler_matches()` — a handler requiring an argument the descriptor does not declare raises `ValueError` at build time. This is refinement 2 of the expert review: the handler and its own descriptor disagreeing is a build defect, not a bad request.
2. Unsupported `ToolParam.type` -> `ValueError` naming the param, rather than silently minting an untyped field.

**Gates:** ruff check + format clean; **16/16** new tests; **36/36** existing contract/router tests (`tests/contract/`, `test_search_contract.py`, `test_agents_router.py`) unchanged; full collection still 46184/46886 with only the intentional `soul_path` error. **Cold-start:** `import infrastructure.contract` = 0.55s wall, of which only **7.8ms** is incremental over `fastapi`/`pydantic` (already imported at boot) — measured, not assumed.

**Mechanism worth remembering (documented in the module docstring):** under `from __future__ import annotations` a closure reference like `q: _model` is stringified and unresolvable by FastAPI. The endpoint therefore declares the param with a default (`payload=Body(...)` / `query=Depends()`) and then assigns `endpoint.__annotations__` with **real objects**. Both variants were probed against a live `TestClient` before committing to one.

**Also surfaced (not yet actioned):** there are **three separate `ToolSpec`/`ToolParam` classes** — `domain/agents/_internal/tools.py` (the descriptor this card means), `domain/tools/_internal/profiles.py` (different signature: `multiline=`, `optional=`), and `packages/avion/src/avion/ai/tools.py`. That is a third copy the AGENTS "two tool tiers, one contract" doctrine does not yet account for; worth a follow-up card, not a change on this one.

**NEXT:** (5) pilot `routers/contracts.py` + `GET /contracts` + a `docs/routers.md` row; (6) `gen_router_manifest.py` / `_manifest.py` **replacing** the `_router_names` loop while preserving deferred imports, per-router `try/except` isolation and registration order; (7) `check_contract.py` + baseline; (8) gates incl. before/after cold-start measurement.

**COORDINATION 2026-10-04 06:16 — PARALLEL IMPLEMENTATION, resolve before merging.** A second session committed `77fbbb817 feat(transport): create_router(spec, route, handler)` on branch `feat/create-router-projection` (root checkout) at 06:02:59 WAT — 54 s after this card's `fa7f62eb6` merged to main and 3 s before the push (`8074dc475`). Different design (`resolve_method` / `build_request_model`, 222-line `contract.py` + `test_contract_projection.py`, `routers/__init__.py` untouched) vs the merged one (`derive_method` + registry + generated `_manifest.py` + contract CI, 483 lines, 35 tests). **Both add `apps/api/server/infrastructure/contract.py`, so that branch will conflict on merge.** Main is the shared copy now: build on `create_router` there (AGENTS "check if it exists / one shared copy"), do not re-implement. Owner of `feat/create-router-projection` should rebase onto `8074dc475`, keep only what the merged design lacks (e.g. `build_request_model` if genuinely different), and retire the duplicate file.

**RECONCILIATION GUIDE 2026-10-04 06:2x (for the `feat/create-router-projection` session).** Their branch now duplicates BOTH halves of this card — `77fbbb817` (`contract.py`, 222 L) and `37112a9f9` (mount table: `_manifest.py` + `gen_router_manifest.py` + `routers/__init__.py` + parity-script edits). Six files collide with what is already on `main` (`7ee2b8c0c`), so the merge is guaranteed to conflict: `infrastructure/contract.py`, `routers/_manifest.py`, `routers/__init__.py`, `scripts/gen_router_manifest.py`, `scripts/check_docs_api_parity.py`, `docs/TRANSPORT_PROJECTIONS.md`.

*Equivalence map (theirs -> already on main):* `resolve_method(spec, override)` -> `derive_method(spec)` + explicit `RouteSpec.method=` that is **logged** (drift visibility is the documented decision); `build_request_model(spec)` -> `_request_model(spec)` (private, JSON-Schema -> Pydantic, 422 contract violations); `check_handler_matches(spec, handler)` -> `_check_handler_accepts(spec, handler)`; `ROUTER_MODULES` tuple -> same symbol, same file, 53 modules in preserved historical order.

*What main has that theirs does not (stop building these):* `ContractRegistry`/`RouteEntry` + `REGISTRY` cross-check, envelope + `auth_scope` + error classification per projected route, `_openapi_extra` (descriptor params published to OpenAPI), the pilot `GET /contracts`, the `contract-gate` CI job (static+runtime blocking, route/**OpenAPI-operation** baseline report-only over 491 operations), docs truth-up (`AGENTS.md`, `TRANSPORT_PROJECTIONS`, `routers.md`, `API.md` 607/58), and 35 passing contract tests. CI proof: `Contract gate` green on both my pushes; `test-server` FAILED-set byte-identical to the pre-change baseline (112 lines, zero mine).

*What theirs has that main lacks — port these instead of re-implementing:* (1) `check_pre_lifespan()` + `_has_module_router()` in `gen_router_manifest.py` — a "every manifest entry actually mounts an HTTP router" assertion main does not have (main excludes main.py-registered modules by parsing `main.py`; theirs verifies mounted-ness — complementary, keep theirs); (2) `tests/server/test_router_manifest.py` (98 L) — keep if it adds cases beyond `test_contracts.py::test_router_manifest_is_fresh`; (3) drop the legacy `_router_names` dual-source regex check — `_router_names` no longer exists on main, so it is dead; (4) resolve `check_docs_api_parity.py` in favour of main's AST version (theirs is regex and misses `RouteSpec(...)` first-arg-less forms).

**Recommended:** `git rebase main`, delete their `contract.py` / `_manifest.py` / `gen_router_manifest.py` (superseded wholesale), cherry-pick only (1)+(2) above, keep main's parity script.

---

**PROGRESS 2026-10-04 (2) — deliverables (1), (2) and (5) LANDED. Branch `feat/create-router-projection`.**

| # | Deliverable | State | Commit |
|---|---|---|---|
| 1 | `create_router(spec, route)` | ✅ | `77fbbb817` |
| 2 | boot-time registration via generated `_manifest.py` | ✅ | `37112a9f9` |
| 5 | pilot `GET /contracts` | ✅ | `5f20dc84b` |
| 3 | CI contract gate (descriptor blocking + OpenAPI diff report-only) | ⬜ next | — |
| 4 | TS projection into `http-client.ts` | ⬜ | — |

**(2) Mount table.** `scripts/gen_router_manifest.py` derives membership from the filesystem (a file qualifies iff it binds a module-level `router`), excludes the 4 pre-lifespan routers with a self-verification against `main.py`, and **reconciles order instead of sorting** — `inference` must precede `chat`, and alphabetical order would have put `chat` first, silently forking OpenAPI from runtime (first-match-wins vs last-wins). Generated manifest reproduces the old list **byte-identically (53/53, same order)**, binds no router modules, and cold start is unchanged (0.43/0.49s vs 0.47s baseline). `routers/__init__.py` keeps deferred imports + per-router `try/except`. Gate: `--check` + `tests/server/test_router_manifest.py`.

**Coupling found and handled:** `check_docs_api_parity.py` (CI) regex-parsed `_router_names` out of `routers/__init__.py`. Deleting it would have made `_mounted_routers()` return `None`, which makes the collision gate *err toward flagging* — i.e. a refactor that quietly turns CI red. The gate now reads `_manifest.py` (legacy table as fallback) and skips `_*` modules so `_manifest.py` is not counted as a 59th router.

**(5) Pilot.** `routers/contracts.py` declares one `ToolSpec`; `create_router` derives everything. `create_router` now records each projection, so `GET /contracts` reads back what projection registered — **a capability registers itself by being projected, never by being added to a list.** `execute` and the HTTP handler are the same callable.

**Style 4 added to the parity gate:** it previously only understood decorators and `add_api_route(...)`, so a projected route was *invisible* — which would make a correct `docs/routers.md` row read as a false "docs claim but code lacks" gap. It now resolves `create_router` the same way `resolve_method` does (idempotent→GET, action→POST, explicit `method=` wins) and marks anything unresolvable **dynamic instead of guessing**, because a wrong verb there would fork the gate from what FastAPI serves.

**Boundary gate caught a real violation:** `tests/contract/test_router_boundary.py` forbids routers importing `domain.*._internal` (baseline may only shrink). Fixed properly — `domain/agents/__init__.py` now re-exports `ToolSpec`/`ToolParam`. Measured: facade **0.335s vs 0.350s** for the private module, i.e. free. **Mistake corrected:** I first bumped `routers_internal_*` claims to 12/37/9 to match — but those claims measure *internal coupling*, and fixing the import dropped the count back to 11/36/8. Claims reverted to untouched; the right fix was the facade, not the ledger.

**Gates at every commit:** 79 tests green (5 pilot + 4 manifest + 16 projection), ruff clean, `gen_router_manifest.py --check` 0, boundary 4/4, parity **6 architecture alarms = exact baseline** (pre-existing `controllers_*`/`combined_*` drift, not mine), other sessions' dirty `.dev-notes` / `.kanban` / `docs/PUGQEEP.md` / `tests/test_soul_path.py` untouched.

**Known pre-existing red (not caused here):** `check_docs_api_parity.py` already exited 1 before this work — 6 arch ALARMs plus the unsectioned `search` router. CI's parity step has no `continue-on-error`.

---

**PROGRESS 2026-10-04 (3) — the permanently-red gate is now GREEN. Commit `7c400b964`.**

`scripts/check_docs_api_parity.py` (a CI step with no `continue-on-error`) had been exiting 1 for a long time. It now exits **0**: 59 routers / 703 routes / 60 sections / 697 rows / 0 missing sections / 0 missing rows / 0 collisions / **0 architecture mismatches** (was 6).

**Inference worth carrying forward:**
1. **Six alarms, one root cause.** The machine-readable `parity-claims` block was never updated when `controllers/` was first brought into scope — it still held pre-scope values `2/9/3` while the prose table *directly beneath it* already read `4/30/13`. Fixing the block cleared all six at once. **When N independent alarms move together, look for the one thing they share before touching any of them.**
2. **Why this was NOT the ledger-fixing error I made earlier.** Discriminator: *did my change move the number?* On `routers_internal_*` yes (my `_internal` import) → fix the code, and the number returned to baseline on its own. Here the number was stale from before `controllers/` was ever counted, nobody's change moved it, and the doc's own contract says "update in the same change that moves the code." Syncing establishes the baseline; the ratchet then only moves **down**.
3. **The real signal was one line, and I named it rather than absorbing it:** prose said `health.py 5`, measured `health.py 6`. `health.py` gained a `domain.*._internal` import *after* the prose was truth-uped. Recorded here so the +1 is not silently blessed.
4. **A red gate is a dead gate.** It had been red long enough that nobody noticed six alarms — which is exactly how the claims drifted to 3x without anyone seeing them. Green gates enforce; red gates train people to skip them.

**COORDINATION (no other card modified; both have `note_hashes=[]` so there was no note to append to):**
- **`b22e878a` (todo)** — "Router/controller/engine conformance — grouped root-cause fixes". Owns the actual reduction (your decision: keep `controllers/` but proxy through engines, seam `router -> controller -> engine -> domain`, GROUP B "make controllers thin proxies, then document the layer honestly"). **It should now *lower* `controllers_internal_*` (4/31/14) and `combined_*` (15/67) as it lands — that is the ratchet working.** It also owns the GROUP D doc-claim rot; I only touched the `parity-claims` block it depends on.
- **`752ca443` (review)** — general search. Its `GET /search` shipped without a `docs/routers.md` row; I added the section (4 lines) to close the gap. Its doc obligation is now met.
- **Card 071 (in_progress, separate worktree)** — `docs/routers.md` truth-up. I touched that file twice this session (Contracts section, Search section); both are additive and will conflict on its regeneration.

**Still open from earlier:** the 2 deadlocked pytest runs (kill/leave undecided), board steps 3+4 (dup-id merge + unique-id assert), options A/B/C on the 55 duplicate basenames.

**PUSH — measured 2026-10-04, NOT a simple push (correcting an earlier "6 commits local-only" note):**
- `origin/main` is **fresh** (fetched 06:24 today) and is 0 ahead of local `main` — no remote-only commits.
- Local `main` has **12 unpushed commits** (other session's landing work: `d5244478d`..`c92adc47b`), none of them mine.
- My 4 commits (`77fbbb817`, `37112a9f9`, `5f20dc84b`, `7c400b964`) are **not in main**.
- `feat/create-router-projection` **does not exist on origin** and diverged from `main` at `f27654e54` (2026-09-25): branch has 254 unique, main has 294 unique. Many pairs are the same change under different SHAs (cherry-picked onto main), so a plain merge would be conflict-heavy.
- Repo's demonstrated pattern is **cherry-pick onto main** (`feat/landing-7-cards`, card `d083e732`) rather than pushing a long-lived diverged branch. That, plus the 12 unpushed main commits, needs an explicit decision — not a unilateral push.

---

**COORDINATION FINDING 2026-10-04 07:20 — the transport card was landed by someone else; cherry-pick aborted.**

Directed to cherry-pick my 4 commits onto `main`, I hit an **add/add conflict on `apps/api/server/infrastructure/contract.py`**: `main` already has it.

**`c38c4e04b`** (2026-10-04, other session) — *"transport: descriptor-projected routes (create_router), boot manifest, contract CI"* — 4007 insertions, 15 files. It is a **superset of all 4 of my commits**:

| | `main` `c38c4e04b` | my commits |
|---|---|---|
| `infrastructure/contract.py` | 483 lines | 222 |
| `gen_router_manifest.py` + `routers/_manifest.py` | ✅ | ✅ |
| `routers/contracts.py` pilot | ✅ | ✅ |
| `domain/agents` facade | ✅ | ✅ |
| **deliverable (3)**: `contract-baseline.json` (2227L) + `check_contract.py` (404L) + `ci_cd.yml` `--static`/`--runtime` gates | ✅ | ❌ *I had recorded it "not started"* |
| tests | **35** (`test_contracts.py`) | 25 |

Their 35 cover everything mine do — verb derivation + explicit override, `x-contract`, shared envelope, 422, handler-error classification, auth scope, manifest freshness, pilot inventory — **plus** query-param coercion, operation-id uniqueness, duplicate method/path rejection, and OpenAPI diff-vs-baseline. Main's `ci_cd.yml` gates on `gen_router_manifest --check`, `check_contract --static`, `check_contract --runtime`.

**=> my `77fbbb817`, `37112a9f9`, `5f20dc84b` are superseded; cherry-pick aborted, nothing applied.** (Root cause of the duplication: the branch forked 2026-09-25 and the same card was worked on both sides. *Rule: check `git log --all -- <path>` for a same-day landing before building on a 9-day-old fork.*)

**THE GATE WORK IS NOT MINE TO LAND — three diverged branches hold fragments of it:**

| branch | state | uniquely holds |
|---|---|---|
| `main` `c92adc47b` | tip | transport ✅; docs **stale** (43 sections/394 rows); **no parity gate, no arch check**; `ci.yml` last touched **2026-09-13** |
| `feat/create-router-projection` (here) | 255 ahead / 294 behind | docs truth-up **60 sections/697 rows** (`d57c68cbc`, card `34b396b7`); arch-contract check (`81616a6f7`); `parity-claims` block (`2d005fd78`); parity wired in `ci.yml:189` (`870ab1bad`); **parity exit 0** |
| `fix/router-engine-conformance` | **207 ahead / 294 behind, stale** | `832ec9e56` *"make doc-vs-code parity an executable alarm and run it in CI"*; `3e4d4e6a4` contract block — **ancestors of neither `main` nor this branch** |
| `.wt-next` `fix/main-gate-green` | tip, **13 dirty files, 25 s old** | actively fixing main's gate **now** — cards `cbd2ffc3` [doing] + `ca647de2` [in_progress] |

**Why the docs cannot simply be ported:** the code diverged. My branch has `collections.py` + `search.py` (main lacks both); main has `doctor.py` (I lack it); `memory` 18v14, `vm` 9v5, `session` 4v3 routes. My 697 rows document *my* code — applied to `main` they would manufacture ~130 dead rows. Main's parity (exit **1**) needs its docs **regenerated against main's own code**: 17 missing sections, 339 undocumented routes, 129 dead rows, 2 untested routers (`model_stack`, `phoneme`). That is card `cbd2ffc3`/`ca647de2`'s job, in flight — I would have duplicated it exactly as the transport card was duplicated.

**Main's parity is red but *unenforced*** — no workflow runs `check_docs_api_parity.py`; only this branch's `ci.yml:189` does. So `7c400b964` made green a gate only this branch enforces.

**Salvageable from my branch for whoever lands it** (none of it duplicated): the `parity-claims` DOWNWARD-ratchet mechanism + arch-contract check (neither exists on `main`), and the `health.py 5→6` undocumented-internal-import signal.

---

**LANDED ON A PREP BRANCH 2026-10-04 07:45 — the one fragment nobody owns. Commit `82c043e61` on `feat/transport-land` (branched from `main` tip `c92adc47b`, worktree `/tmp/opencode/wt-land`).**

Scope: `parity-claims` DOWNWARD-ratchet block + arch-contract check + `--architecture-only` + 7 tests. 3 files, +352. **No route docs touched** (that is card `cbd2ffc3`), **no CI wiring touched** (that is card `ca647de2`). Verified: `--architecture-only` exit 0 (11 claims / 0 mismatched / 0 unresolved); tamper 30→31 exits 1 with `ALARM controllers_internal_stmts: doc=31 code=30`; full parity unchanged; ruff clean; 7 passed.

**TWO MEASUREMENTS THAT CORRECT THE PREVIOUS ENTRY — read these before trusting it:**

1. **`main` measures `controllers_internal 4/30/13` — exactly what the prose table already claims.** The `4/31/14` I recorded earlier as "prose was truth-uped, code drifted +1" is drift **on `feat/create-router-projection`**, not on `main`: `health.py` is **5** on main, **6** on my branch. So my `7c400b964` synced the claims to **bless my own branch's drift** rather than exposing it. The prose (4/30/13) was right; the measurement was the outlier. *Rule: when a number disagrees with a prose claim, first check whether the prose matches some OTHER clean tree — divergence in one branch is not global truth.*
2. **`main` reaches `domain.*_internal` from only `1` router / `2` stmts — `doctor.py` alone.** My branch: **11 files / 36 stmts**. Main has landed facade/conformance work my branch lacks (its `controllers_lines` 2844 vs my 3223, `controllers_defs` 93 vs 105). `doctor.py` exists **only on main**; `collections.py` + `search.py` exist **only on my branch**. **Main is ahead on conformance; my branch is ahead on docs.** Neither branch can be landed whole.

**Design decision:** `--architecture-only` exists so the contract is enforceable *independently* of the route-doc truth-up (G1). Without it the check could not go into CI until the docs landed — which is precisely how it stayed unenforced the first time. The drift test monkeypatches `CLAIMS_MD` at a tmp file instead of editing the tracked audit doc, so a failed assertion cannot leave the repo modified.

**Pre-existing, NOT mine, still open:** `main`'s `tests/` collection has **3 errors**, all `ModuleNotFoundError: No module named 'mogdb'` (`tests/server/test_audit_instrumentation.py`, `test_auth_router.py`, `test_datasets_router.py`). Root cause: `packages/mogdb` is **src-layout** (`src/mogdb/`) and is only `pip install -e packages/mogdb`'d in CI's dedicated `mogdb` job — the general env never installs it, and main recently imported it into these paths. Adding `packages/mogdb` to `PYTHONPATH` does **not** fix it (src-layout). Belongs to **`ca647de2`** (CI red on main since 10-03). 2970 tests collect otherwise; my test file collects clean.

**Pending:** landing `82c043e61` needs a coordination window — `main` is checked out at `/home/mana/Documents/sloughgpt-api-std`, and `.wt-next` (`fix/main-gate-green`) was still active as of 07:40 (`doctor.py` dirty).

---

**APPENDIX 6 — INCIDENT 2026-10-04 08:27: lint-staged stranded the other session's work (RESOLVED, stash left for them).**

While committing a `.py` fix on `feat/create-router-projection`, husky fired `npx lint-staged`, which **stashes the whole worktree before running tasks**. The task then died with `ENOENT`, so the stash was never dropped and five files were stranded:

`.dev-notes/notes.journal.jsonl` (19 note records), `.kanban/board.jsonl` (2 cards — `0d4ad67b`, `5cee61f6`), `.kanban/slot_history.jsonl` (3 entries), `docs/PUGQEEP.md` (7 lines of the Pool/`_stems` lock analysis), plus my own B011 edit.

**Three-layer root cause — note that two of these are separate defects:**

1. **`acfb08870` (2026-09-13, "fix: use venv ruff in lint-staged hook")** pointed this branch's `package.json` `*.py` rule at `.venv/bin/ruff`.
2. **`.venv` does not exist** — in any of the four trees (repo, `wt-land`, `sloughgpt-api-std`, `.wt-next`). It is gitignored, so it is environment-local and gone. AGENTS.md still describes it as extant.
3. **`find_server_python()` in `domain.shared` implements only `.venv -> sys.executable`.** `scripts/python`'s header advertises *"conda env `sloughgpt` -> `.venv` -> the interpreter doing the query"* — **the conda check it documents does not exist.** So main's own rule (`scripts/python -m ruff`) *also* fails here: it resolves to base `miniconda3/bin/python3`, which has no ruff. **Both branches' hooks were broken; they just failed differently.**

**Fixed:**
- `b59020473` (other session, 09:25) — rule now runs `/home/mana/miniconda3/envs/sloughgpt/bin/ruff` directly. This is the "use conda" resolution.
- `f1ed4a9e5` — my B011 fix (`assert False` -> `raise AssertionError`, so the guard survives `python -O`). **First commit to exercise the new hook: passed, and all four of their dirty files were restored intact.**
- `80950c63a` — restored `scripts/python`, which this branch had been missing since the 2026-09-25 divergence (489 files reference it). Copied verbatim from `c92adc47b`, no edits.
- `db621f5c0` on `feat/transport-land` — `zzz_test.py` deleted (F822: `__all__ = ["ZZZ"]`, accidentally committed by `89e01b58c`) + `tests/test_domain_print_allowlist.py`, an allowlist ratchet over `domain/`'s 92 legitimate `print()` calls. ruff green there.

**STRANDED, LEFT FOR THE OTHER SESSION (per user decision): `stash@{0}` (94d6917d4).** Not applied, not dropped. A **verified** independent copy is at `/tmp/opencode/stash_rescue` (5343 files; confirmed to contain all 4 sample stranded note IDs, both card IDs, the PUGQEEP prose, and my original edit). A second insurance snapshot of their four live files taken before committing is at `/tmp/opencode/precommit_backup`. `docs/PUGQEEP.md` is still the only stranded content that is *not* recoverable from the worktree itself.

**Trap that made this hard to diagnose, recorded so it is not repeated:** three separate measurements this session reported a false clean result because the exit code was read from the *last* command in a pipe (`grep ... | sed ... || echo`) — `sed` succeeds even when `grep` finds nothing. `rg` also is not installed, so an earlier `2>/dev/null` swallowed `command not found` and produced four zeros that looked like a clean scan. **Never measure a gate through a pipe, and never suppress stderr on a first-time tool probe.**

**Still open / needs a decision:**
- `find_server_python()` gap — fixing it changes which interpreter server startup and tests resolve to. Own review, no ride-alongs.
- `AGENTS.md` still describes `.venv` as extant; it is absent everywhere.
- My branch's remaining work is still unlanded: `82c043e61` + `db621f5c0` sit on `feat/transport-land`, awaiting a coordination window for `main`.

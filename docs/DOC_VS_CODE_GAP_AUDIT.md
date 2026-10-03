# Doc-vs-code gap audit — API surface

> Ask 2026-08-29: "compare old+updated docs vs current API implementations,
> cross-reference test results, produce gap report -> cards."
> Delivered 2026-10-01 [card 20260929_083]. Re-run any time:
> `python scripts/check_docs_api_parity.py` (also `--json` / `--markdown`;
> exit 1 while gaps remain). CI runs it on every PR and push to main as the
> **Doc/code parity** job, so these gaps now fail the build instead of being
> noticed later.

## Method

Three sources, cross-referenced mechanically:

1. **Code truth** — route registrations in `apps/api/server/routers/*.py` + `main.py`
   - `training/router.py` (mounted via `infrastructure/startup.py`), covering all
     three registration styles in the codebase: class-based
     `self.router.add_api_route(path, handler, methods=[...])` (the dominant one,
     600+ calls), module-style `@router.get(...)`, and `@app.get(...)` — each joined
     with its `APIRouter(prefix=...)`.
2. **Doc claims** — endpoint tables in `docs/routers.md`, count claims in
   `docs/API.md`, page/router counts in `docs/PRODUCT_ENGINEERING.md`.
3. **Test coverage** — per-router file coverage across `tests/server/` and
   `apps/api/server/tests/`. (Runtime ping coverage exists separately:
   `tests/server/test_endpoint_registry.py`, deselected `-m slow`.)

Sample dead rows were grep-verified by hand: `/session/list`, `/models/huggingface`
and `/operations/cancel-all` have **zero** route hits anywhere under
`apps/api/server/` — these are real doc rot, not parser noise.

## Measured vs claimed

| What                       | Docs claim                      | Code says                                                         | Verdict |
| -------------------------- | ------------------------------- | ----------------------------------------------------------------- | ------- |
| HTTP routes                | 412 (`API.md`)                  | **619** literal + 1 dynamic                                       | stale   |
| Router count               | 43 (`API.md`) / 58 (`PROD_ENG`) | **57** files, all mounted except `api_keys` (unmounted by design) | stale   |
| `routers.md` endpoint rows | 398 rows / 42 sections          | **120** rows match no code route                                  | drift   |
| Frontend pages             | 85+ (`PROD_ENG`)                | **109** `page.tsx`                                                | stale   |

## Gaps → cards

### G1 — API docs truth-up (card `34b396b7`)

- **120 dead rows** in `docs/routers.md` claim endpoints code no longer has.
  Samples: `GET /session/list`, `POST /session/create`, `GET|DELETE
/session/{session_id}`, `/session/{session_id}/suggestions`,
  `POST /context/store-fact`, `POST /operations/{op_id}/cancel`,
  `POST /operations/cancel-all`, `GET /models/huggingface`,
  `GET /models/download/status`, `POST /models/download/cancel|retry`.
- **16 mounted routers have no section at all**: `chat`, `cloud_training`,
  `consciousness`, `dashboard`, `mobile`, `model_stack`, `openwebui`, `phoneme`,
  `plugins`, `profiles`, `settings`, `tenants`, `tokens`, `tools`, `users`,
  `workspaces`.
- **339 code routes have no endpoint row** (top: settings 41, consciousness 35,
  kb 34, mobile 33, workspaces 29, models 25, inference 19).
- Stale counts to re-pin: `API.md` "412 routes across 43 routers";
  `PRODUCT_ENGINEERING.md` "85+ frontend pages" (109) and "58 routers" (57).

Full lists come from `check_docs_api_parity.py --json`; the fix can regenerate
rows from that output instead of hand-editing.

### G2 — backend test coverage for 3 routers (card `f71f9f9e`)

| Router           | Surface                                                                    | Lines | Coverage today                                                              |
| ---------------- | -------------------------------------------------------------------------- | ----- | --------------------------------------------------------------------------- |
| `chat.py`        | `POST /chat`, `/chat/stream`, regenerate, cancel, GET active/health/addons | 246   | none for the HTTP router (existing `test_chat_*` cover manager/trainer/CLI) |
| `model_stack.py` | model-stack routes                                                         | 116   | zero tests anywhere                                                         |
| `phoneme.py`     | phoneme router                                                             | 158   | frontend only (`phoneme-controller.test.ts`)                                |

## Out of scope (deliberately)

- Route Map per-row merge statuses (⏳/🔧) — the frontend lane is rotating
  branches; recount after that merge work lands.
- Per-method SDK coverage columns in `API.md` — only the count claims were
  checked, not SDK bindings.
- Making `routers.md`/`API.md` **generated** from the parity script — proposed
  as the fix mechanism inside card `34b396b7`, not done here.

## Not a duplicate

- `tests/test_shared_indexes.py` guards doc/app/package index registration only.
- `tests/server/test_endpoint_registry.py` pings live routes at runtime (slow).
- Nothing compared **doc claims ↔ code routes** statically until this script.

---

## Re-run — 2026-10-01 (scope widened past API surface)

Evidence pack for the product/engineering review. House rule for this section:
every number comes from a run made today, never from a remembered or previously
written figure.

### API surface — now clean

`PYTHONPATH=. python scripts/check_docs_api_parity.py --markdown` → **exit 0**:

- **57** router files, **703** literal routes (1 dynamic, not diffed)
- `docs/API.md` claim `(703, 57)` → **ok**
- `docs/routers.md`: 58 sections / 697 rows → **0** drift, **0** unsectioned
  mounted routers, **0** code routes without a row
- **0** routers without a test file

G1 and G2 above are therefore **closed**. The _Measured vs claimed_ table is now
the pre-fix state (619 routes, 120 dead rows, 339 undocumented) — do not read it
as current.

> **Counting caveat.** Decorator counting under-reports badly: the dominant
> idiom is class-based `self.router.add_api_route(...)` (~620 calls) while
> decorator style is 95. Counting `@router.get(` yields 17 for `routers/`; the
> script's 703 is the number to trust.

### Doc claim-rot — 8 claims, all disproven against code

| Doc                           | Claim                                            | Code says                                                                                                                                       |
| ----------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `ROADMAP.md` g7               | Voyager "201 tests, 7 backends"                  | `packages/voyager/` is only `pyproject.toml`+`src` — **0 test files**                                                                           |
| `ROADMAP.md` g8               | journeys run "using Voyager"                     | `scripts/run_journey_tests.py` runs **Arken**+Playwright (8 journeys, names correct)                                                            |
| `ROADMAP.md` g10              | `_get_trainer()` 500s on `/status`, `/train/*`   | `domain/consciousness` deleted, **zero** live `domain.consciousness.*` imports; router does `from domain.cognition import ConsciousnessTrainer` |
| `PRODUCT_ENGINEERING.md` r15  | consciousness "not wired"                        | `ConsciousnessProcessor` exported from `domain/models/__init__.py`                                                                              |
| `PRODUCT_ENGINEERING.md` r16  | voice "router bypasses domain"                   | **0** `_internal` in `voice.py`/`phoneme.py`                                                                                                    |
| `PRODUCT_ENGINEERING.md` l319 | `_internal` "remains only in `shell.py`/`vm.py`" | **11** routers import `domain.*_internal.*`                                                                                                     |
| `USER_JOURNEYS.md`            | results → `tests/test_results/`                  | code writes `packages/core-py/tests/test_results/`                                                                                              |
| this document, above          | G1/G2 open                                       | parity script **exit 0**                                                                                                                        |

Seven of eight resolve the same way — **the docs lag the code**. The exception
is the `_internal` row, where the code is _worse_ than the doc claims.

### Router → engine conformance (measured)

<!-- parity-claims:v1
# Contract parsed by scripts/check_docs_api_parity.py. If measured code
# disagrees with any value below, that script exits 1 (alarm). Update this
# block in the SAME change that moves the code.
routers_internal_files=11
routers_internal_stmts=36
routers_internal_in_scope_files=8
controllers_files=5
controllers_lines=3215
controllers_defs=105
controllers_internal_files=2
controllers_internal_stmts=9
controllers_internal_modules=3
combined_internal_files=13
combined_internal_stmts=45
-->

| Category                                         | Count / 57                               | Note                                                                                                                                                             |
| ------------------------------------------------ | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| reaches core directly, 0 `_internal`             | 37                                       |                                                                                                                                                                  |
| reaches core via `controllers/`                  | +1 (`feedback`)                          | `controllers/feedback.py` holds 6 domain imports                                                                                                                 |
| imports `domain.*_internal.*`                    | **11** routers / **36** statements       | 8 routers in scope; `shell`/`vm`/`world_render` sit in a concurrent-refactor zone                                                                                |
| **`controllers/` imports `domain.*_internal.*`** | **4 files / 30 statements / 13 modules** | `models.py` 12, `datasets.py` 8, `feedback.py` 5, `health.py` 5, `config.py` 0 — **counted here for the first time; the original audit only scanned `routers/`** |
| never touches core                               | 6                                        | `images`, `docstore`, `api_keys`, `security`, `ratelimit`, `config` — all infra/security                                                                         |

**Combined: 66 `_internal` import statements across 15 files.** Worst targets:
`domain.models._internal.provider` ×9, `domain.training._internal.cache_tags` ×7,
`domain.infrastructure._internal.artifact_registry` ×2.

The Router Playbook only forbids `_internal` in **routers**, so nothing in the
documented rules covers the 30 statements in `controllers/`. An executable
conformance check has to span both directories.

> Also in scope: `controllers/` is **directly tested** —
> `test_{health,models,config,datasets}_controller.py` patch controller symbols.
> Playbook rule 8 says tests must patch the facade, never module symbols, so
> those 4 files get re-pointed when controller logic moves into an engine.

The playbook's literal `grep -c _internal <router> = 0` flags **13**: two are
**false positives** (`tokenizer.py:6`, `cloud_training.py:4` mention the string
in a docstring asserting they do _not_ import it). An executable check must
parse imports rather than grep the token.

### Refresh — 2026-10-02 (the numbers moved)

Re-measured the next day. This table _is_ the argument for an executable check
rather than prose:

| Claim (2026-10-01)                                          | HEAD `aed61d144` (2026-10-02)     | Verdict      |
| ----------------------------------------------------------- | --------------------------------- | ------------ |
| routers: 11 files / 36 statements                           | **11 / 36**                       | ✅ unchanged |
| routers in scope: 8                                         | **8**                             | ✅ unchanged |
| `controllers/` `_internal`: 4 files / 30 stmts / 13 modules | **2 files / 9 stmts / 3 modules** | ❌ moved     |
| `controllers/` size: 2,892 lines / 94 defs                  | **3,215 / 105**                   | ❌ moved     |
| combined: 66 stmts / 15 files                               | **45 / 13**                       | ❌ moved     |

Two causes:

- **`3f2378ccb` (2026-09-29)** moved `feedback`/`health`/`models` onto facades —
  `feedback` 5→0, `models` 12→0, `health` 5→1. It had already landed _before_
  the 2026-10-01 figures were taken, so those were read from the **worktree**,
  not from HEAD. The router half of the audit was unaffected, which is why only
  the controller numbers drifted.
- **`56f17ab7b` (2026-10-02)** grew `health.py` 763 → 1,111 lines.

**Uncommitted drift — flagged, not touched (2026-10-02):** the worktree sits at
**31** `_internal` while both HEAD and the staged index sit at **9**. The
worktree edits convert facades _back_ to `_internal`:

```diff
- from domain.models import get_provider
+ from domain.models._internal.provider import get_provider
- from domain.infrastructure import get_lifecycle_manager
+ from domain.infrastructure._internal.lifecycle import get_lifecycle_manager
```

`git commit` (index only) preserves the facade work; `git commit -a` would
silently revert it. Belongs to another session — left alone.

The executable check reports exactly that split, which is the behaviour wanted:

| tree                      | exit            | why                                                                                                                                                  |
| ------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| HEAD — what CI checks out | **0**           | matches the contract                                                                                                                                 |
| index (staged)            | **0**           | matches the contract                                                                                                                                 |
| worktree                  | **1**, 6 alarms | `controllers_internal_stmts` 9 → 31, plus `controllers_lines` 3215→3223, `internal_files` 2→4, `internal_modules` 3→14, `combined_*` 13→15 and 45→67 |

So a local `python scripts/check_docs_api_parity.py` exits 1 in this worktree
today, and that is the check working rather than the check being broken: every
alarm it raises is the uncommitted reversion above, and it would fire the same
way on a `git commit -a`. Verified by running it against three trees — HEAD
archive, index archive, worktree — where only the worktree fails. CI never sees
this, because CI checks out committed content, and the worktree goes green the
moment the reversion is either restored or committed deliberately.

### Undocumented `controllers/` layer

`controllers/{models,health,datasets,feedback,config}.py` = **2,892 lines /
94 defs** as measured 2026-10-01 (**3,215 / 105** at HEAD the following day —
see _Refresh_ above; the per-controller table below is the same 10-01
snapshot). `grep -l controllers` across `PRODUCT_ENGINEERING.md`,
`PYTHON_FIRST.md`, `STRUCTURE.md`, `routers.md` → **no matches**. The Router
Playbook documents `router → engine → domain`; the code runs
`router → controllers → domain`.

What each one actually does (measured 2026-10-01 — none is a pass-through):

| Controller    | Lines | Methods | `_internal` imports | Role                                                                                                                     |
| ------------- | ----- | ------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `config.py`   | 43    | 4       | 0                   | `get/update_generation_config` — **never touches domain at all**                                                         |
| `feedback.py` | 341   | 19      | 5                   | conversation CRUD + `record_feedback` (67 lines) + `_wire_model`                                                         |
| `datasets.py` | 661   | 21      | 8                   | `list_datasets` 129 lines/84 branches, stats 70/39, preview 58/45, versioning, export                                    |
| `health.py`   | 763   | 24      | 5                   | read-only aggregation across 12 domain sources                                                                           |
| `models.py`   | 1,067 | 26      | 12                  | model lifecycle: `_load_hf_model` 146 lines, `load_model_path` 112, `unload_model` 91, device + process-guard management |

Because they own real logic, the fix is **absorption, not deletion**: the
decision on file is that **the engine _is_ the controller** — the responsibility
(moving and serving data) is integrated into an engine until no controller
stands alone in the logic. Shape: `[router + proxy] → engine (controller inside)
→ domain`.

### Performance (`data/benchmark_results/`)

| kind      | runs        | numbers                                                                                                        |
| --------- | ----------- | -------------------------------------------------------------------------------------------------------------- |
| latency   | 2 (1 empty) | Qwen2.5-0.5B mean **31,739 ms**, p50 27,995, p95 **57,772** (n=15)                                             |
| startup   | 8           | `time_to_health` **7.9–152.2 s**; `time_to_ready` up to **230.7 s**, past its own 180 s `api_starting_timeout` |
| execution | 1           | dispatch **4.31 µs**, peak_threads 5                                                                           |
| stability | 1           | overall **90**, crash 0, response 1.0, length_cv 0.80 (n=20)                                                   |
| training  | **0**       | `ROADMAP.md` g16 names this the release gate — nothing recorded                                                |

**The reporter itself was broken — fixed 2026-10-02**
(`fix/benchmark-history-reporter`): `history --kind latency` raised
`AttributeError: 'list' object has no attribute 'get'` because one stored
record holds compression samples as a _list_ rather than a dict; `startup`
and `execution` rendered `mean=?` / `p95=?` because neither ever stores
`mean_ms`; and `history` with no `--kind` listed only stability and latency,
**hiding 9 of the 12 runs on disk**. Each kind now renders the keys it
actually stores, and an unexpected payload is described rather than printed
as `?`:

```
── latency: 2 runs ──
  2026-09-24T04:04:57  served    list(5) non-comparable samples
  2026-08-03T04:12:01  Qwen/…    mean=31739 p50=27994.7 p95=57771.8
── startup: 8 runs ──
  2026-09-25T07:58:49  served    health=152.23 ready=230.72 [past 180s timeout]
```

That last line _is_ the 230.7 s vs 180 s finding in the table above — the
reporter now surfaces it instead of leaving it to be dug out of the JSON.

### Journeys

`packages/core-py/tests/test_results/user_journey_results.json`
(2026-09-29 12:07): **46/46 passed** — but **33 are presence checks**
(19 `nav_*`, 6 `*_loads`, 8 `redirect__*`); only 3 are genuine end-to-end
(`chat_type_message`, `datasets_kaggle_import_success`,
`training_job_detail_back_link`). `comprehensive_training_journey_results.json`
is `[]`. `USER_JOURNEYS.md` still lists ~40 untested routes.

The feature inventory is not unified: `PRODUCT_ENGINEERING.md` declares 20
features, `UX_FLOWS.md` declares 12 flows, `FEATURES.md` cuts it differently
again (dataset + 26 consciousness pages + 7 tools). There is no single count of
"our features".

### Integration — does the handler layer actually import? (added 2026-10-02)

The question behind this row: do the `router → engine → domain` import chains
resolve?

**Method, reported because three attempts were wrong first.** Static import
resolution was tried three ways and disagreed with reality every time, so none
of its numbers are cited:

| static attempt                                    | it claimed                             | ground truth |
| ------------------------------------------------- | -------------------------------------- | ------------ |
| resolve every `from domain… import …` in handlers | 114 missing modules, 120 missing names | all import   |
| same, module-level imports only                   | 19 names, 5 modules broken             | **0** broken |

Four independent reasons, any one of which is fatal: `infrastructure.auth` and
`training.jobs` written inside a router resolve relative to `apps/api/server/`,
not the repo root; the `domain/infrastructure/*.py` facades are pure
star-forwarders (`from domains.… import *`), so no downstream name survives
without expanding them; `domain/inference/__init__.py` exports lazily through
PEP 562 `__getattr__`; and optional dependencies sit behind
`except ImportError`. Grepping instead produces different wrong answers again
— see the tokenizer docstring case in the parity check.

So the measure is the only one that cannot lie: import each module, in its own
subprocess so one hang cannot poison the rest.

| state                            | modules                         | import OK | failed | median import |
| -------------------------------- | ------------------------------- | --------- | ------ | ------------- |
| worktree — the running code      | 63 (58 routers + 5 controllers) | **63**    | 0      | 0.5 s         |
| index — what is staged to commit | 62 (57 routers + 5 controllers) | 61        | **1**  | 0.6 s         |

**The staged tree is worse than the working tree.** `routers.model_stack`
imports fine from disk and fails from the index:

```python
# worktree — resolves
from domain.training._internal.cache_tags import get_cache_root
# index — ImportError: cannot import name 'get_cache_root' from 'domain.training'
from domain.training import get_cache_root
```

The move to the facade form is right; the other half of it never arrived.
`domain/training/__init__.py` exports no `get_cache_root` in HEAD, in the index
or in the worktree, is not in the staged set at all, and the definition sits at
`domain/training/_internal/cache_tags.py`. **One export, staged alongside the
router change, closes it** — and this is one class, not one bug: whenever a
handler switches to `from domain.<f> import <name>`, the export lands in the
same change.

Same measurement, second finding: **`apps/api/server/routers/search.py` exists
only in the worktree, never in the index** — 58 routers on disk, 57 tracked.
That alone explains the count disagreement reported above: `docs/API.md`
claims 702/58, a fresh checkout — what CI actually sees — measures 701/57, and
the worktree measures 702/58 because it holds the untracked file. The count was
regenerated from the worktree, so it encoded one session's uncommitted state.
Until the search feature's owner adds the file, CI will report `STALE` as a
warning (deliberately not a gap), and stale counts can then be promoted to a
failure.

### Group F — duplicate code, no single home (added 2026-10-01)

Same root-cause class as the `controllers/` layer: code lives in more than
one place because no rule says where it belongs. Measured against
`domain/{cognition,core,cognitive}/_internal/`:

| Package            | py files | refs   | Role                                      |
| ------------------ | -------- | ------ | ----------------------------------------- |
| `domain/cognition` | 23       | **77** | live feature slice — the one that is real |
| `domain/core`      | 15       | 20     | CCGT "Core" home; holds duplicates        |
| `domain/cognitive` | 10       | **1**  | stray clone; not the port target          |

**10 files are byte-identical across `cognition/_internal/` and
`core/_internal/`** — 10 identical, **0 differing**, 12 files exist only in
`cognition`. `cognitive` carries a 9-file copy of that same set:

```
base.py  core.py  grounding.py  knowledge_graph_v2.py
metacognition/__init__.py  processor.py
reasoning/{__init__,advanced,deep}.py
```

| Measure                | Bytes                 |
| ---------------------- | --------------------- |
| one copy (9 files)     | 167,376               |
| held by three packages | 502,128               |
| **redundant**          | **334,752 (327 KiB)** |

Only `__init__.py` differs per package (2,887 / 4,312 / 770 bytes); every
implementation file is a literal copy, not a re-export — so a fix to
`reasoning/deep.py` must be applied in three places and nothing enforces it.

**CCGT port state** (Cognitive / Core / Gateway / Training — "the only four
domains", `PRODUCT_ENGINEERING.md:28`): `training` (380 refs) and `cognition`
are live; **`domain/gateway/` does not exist** (only `apps/gateway/`, the edge
relay); legacy `packages/core-py/domains/` still holds **268 py files across
36 subpackages**, reached by 45 star-shims in `domain/`.

Remedy is already stated in `ROADMAP.md` ("old files deleted — no deprecated
shims"): one home, others import it.

> **Measurement note:** two earlier checks reported 7 identical and then 0/10 —
> both were wrong, caused by building doubled `_internal/_internal/` candidate
> paths. The numbers above come from a run that prints every candidate path, so
> each `CLONE` is auditable.

### Test status

- **Web: 840 files / 8,169 tests, all passing** (2026-10-01).
- Python suite: pending this run — fill in before citing.

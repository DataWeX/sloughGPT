# Doc-vs-code gap audit — API surface

> Ask 2026-08-29: "compare old+updated docs vs current API implementations,
> cross-reference test results, produce gap report -> cards."
> Delivered 2026-10-01 [card 20260929_083]. Re-run any time:
> `scripts/python scripts/check_docs_api_parity.py` (also `--json` / `--markdown`;
> exit 1 while gaps remain).

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
| HTTP routes                | 607 (`API.md`, re-pinned)       | **607** literal + 1 dynamic                                       | ok      |
| Router count               | 58 (`API.md` + `PROD_ENG`)      | **58** files, all mounted except `api_keys` (unmounted by design) | ok      |
| `routers.md` endpoint rows | 394 rows / 43 sections          | **129** rows match no code route                                  | drift   |
| Frontend pages             | 85+ (`PROD_ENG`)                | **108** `page.tsx`                                                | stale   |

*(re-measured 2026-10-09 on main `cd9db5d35` via `scripts/check_docs_api_parity.py`;
the `routers.md` drift grew 120→129 as routes changed — G1 owns the regeneration.)*

### Machine contract (`parity-claims`)

The prose table above rots silently. The block below is the machine-readable
form of the same idea, parsed by `scripts/check_docs_api_parity.py`: if
measured code disagrees with any value, the script exits 1.

<!-- parity-claims:v1
# Contract parsed by scripts/check_docs_api_parity.py. If measured code
# disagrees with any value below, that script exits 1 (alarm). Update this
# block in the SAME change that moves the code.
#
# 2026-10-04 baseline, measured on main at c92adc47b; re-measured when main
# advanced to 2d4a7118d; architecture claims re-verified 2026-10-09 on main
# cd9db5d35 (11 claims, 0 mismatched). The routers_* values fell 1->0 files,
# 2->0 stmts
# there because doctor.py moved onto the domain.core facade
# (domain.core._internal.doctor.* -> domain.core.*), which is the conformance
# work this ratchet exists to record. Before lowering a value, confirm the
# cause by diff and check the facade still re-exports the names (domain.core
# exports run_doctor and default_report_path from _internal) — a drop traced
# to a real facading change is the ratchet working, not measurement drift.
# Once in sync these act
# as a DOWNWARD ratchet: the router/controller conformance work (card
# b22e878a) lowers them as it lands, and any climb shows up as an alarm.
# Check independently of the route-doc truth-up (G1) with:
#   scripts/check_docs_api_parity.py --architecture-only
routers_internal_files=0
routers_internal_stmts=0
routers_internal_in_scope_files=0
controllers_files=5
controllers_lines=2844
controllers_defs=93
controllers_internal_files=4
controllers_internal_stmts=30
controllers_internal_modules=13
combined_internal_files=4
combined_internal_stmts=30
-->

## Gaps → cards

### G1 — API docs truth-up (card `34b396b7`)

- **129 dead rows** in `docs/routers.md` claim endpoints code no longer has
  (120 at the 2026-10-04 measurement; 129 re-measured 2026-10-09).
  Samples: `GET /session/list`, `POST /session/create`, `GET|DELETE
/session/{session_id}`, `/session/{session_id}/suggestions`,
  `POST /context/store-fact`, `POST /operations/{op_id}/cancel`,
  `POST /operations/cancel-all`, `GET /models/huggingface`,
  `GET /models/download/status`, `POST /models/download/cancel|retry`.
- **17 mounted routers have no section at all** (16 at the 2026-10-04
  measurement; `mole` added): `chat`, `cloud_training`,
  `consciousness`, `dashboard`, `mobile`, `model_stack`, `mole`, `openwebui`, `phoneme`,
  `plugins`, `profiles`, `settings`, `tenants`, `tokens`, `tools`, `users`,
  `workspaces`.
- **339 code routes have no endpoint row** (top: settings 41, consciousness 35,
  kb 34, mobile 33, workspaces 29, models 25, inference 18).
- Counts to re-pin: ~~`API.md` "412 routes across 43 routers"~~ (re-pinned to
  607/58, now in parity); `PRODUCT_ENGINEERING.md` "85+ frontend pages" (108).
  ("58 routers" in both docs now matches the measured 58.)

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

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
| HTTP routes                | 412 (`API.md`)                  | **619** literal + 1 dynamic                                       | stale   |
| Router count               | 43 (`API.md`) / 58 (`PROD_ENG`) | **57** files, all mounted except `api_keys` (unmounted by design) | stale   |
| `routers.md` endpoint rows | 398 rows / 42 sections          | **120** rows match no code route                                  | drift   |
| Frontend pages             | 85+ (`PROD_ENG`)                | **109** `page.tsx`                                                | stale   |

### Machine contract (`parity-claims`)

The prose table above rots silently. The block below is the machine-readable
form of the same idea, parsed by `scripts/check_docs_api_parity.py`: if
measured code disagrees with any value, the script exits 1.

<!-- parity-claims:v1
# Contract parsed by scripts/check_docs_api_parity.py. If measured code
# disagrees with any value below, that script exits 1 (alarm). Update this
# block in the SAME change that moves the code.
#
# 2026-10-04 baseline, measured on main at c92adc47b. Once in sync these act
# as a DOWNWARD ratchet: the router/controller conformance work (card
# b22e878a) lowers them as it lands, and any climb shows up as an alarm.
# Check independently of the route-doc truth-up (G1) with:
#   scripts/check_docs_api_parity.py --architecture-only
routers_internal_files=1
routers_internal_stmts=2
routers_internal_in_scope_files=1
controllers_files=5
controllers_lines=2844
controllers_defs=93
controllers_internal_files=4
controllers_internal_stmts=30
controllers_internal_modules=13
combined_internal_files=5
combined_internal_stmts=32
-->

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

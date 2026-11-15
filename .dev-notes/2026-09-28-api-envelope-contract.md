---
id: 20260928_070_api_standardization
status: done
title: "Card 070 — Standardize our APIs: pin envelope contract, truth-up API.md"
tags: [backend, api, contract, docs]
sprint: current
gh: null
---
# Card 070 — API standardization

## What we found (measured, not assumed)

- The standard already existed: `schemas/common.py` `StandardResponse` +
  `success_response()` + `error_response()` + `@endpoint()` + `classify_and_raise`.
  55/59 routers used it; `http-client.ts:728` unwraps iff `status`+`data` present.
- **Two sanctioned styles, not one**: envelope (majority) + **typed** — routes
  registered with `response_model=X` return the bare typed payload
  (`GET /knowledge` → `list[KnowledgeItemOut]`, `POST /shell/exec` →
  `ShellExecResponse`, `/inference/*` → `InferResponse`). Frontend + tests
  consume them bare end-to-end. Earlier instinct to "fix shell/infer/kb" was
  wrong — they conform under style 2; only the docs didn't know it.
- Real gaps: `docs/API.md` error taxonomy documented a shape nothing implements
  (`{status:"error", error:{...}}` vs actual flat `{error, code, details?,
  correlation_id?}`), claimed `tests/contract/` pinned shapes (it didn't),
  `/knowledge` example wrong, route counts stale (412/43 vs 617/56), hedged
  "bare responses pass through unchanged"; zero enforcement existed.
- AST scan (registration-aware: decorators + `add_api_route`, pruned to
  handler-own body, response_model-aware): **0 violations** across 617 routes
  (1 bare typed return = kb list, sanctioned; 611 other returns enveloped/Response).

## Changes (merged main `254615744`)

1. `tests/contract/test_api_envelope.py` (new, 14 tests) — shape pins for
   success/error builders (error must be flat, forbids the stale doc shape),
   response_model-aware router return scan, zero-tolerance, anti-vacuous
   `routes_seen > 400` guard.
2. `docs/API.md` — two-style Response Envelope section, flat error taxonomy with
   real example + client pointer, contract-test reference now true,
   617 routes / 56 routers.
3. `tests/contract/test_router_boundary.py` — stale 56-entry
   `_ROUTER_INTERNAL_IMPORTS` baseline shrunk to `frozenset()` (measured 0
   post-card-060).
4. `apps/api/server/routers/kb.py` — `list_knowledge` annotation `-> dict` →
   `-> list` (it returns a typed list; response_model governs, cosmetic).

## Gates

- `ruff check .` clean (repo-wide).
- `tests/contract/` 14 passed.
- `tests/server` 1892 passed / 4 failed — exact pre-existing baseline
  (test_system_router stream ×4, fail on main too).
- root `tests/ --ignore=tests/server` 1145 passed / 9 failed / 5 errors —
  exact pre-existing baseline.

## Follow-ups (noted, not blocking)

- `docs/routers.md` (674 lines) not audited for the same staleness.
- SDK method counts in API.md (84/27/82) unverified.
- `mobile.py` shared-core violation (API.md rule 1) still "being re-pointed".
- 38 routers without `response_model` → incomplete OpenAPI (deferred; no
  consumer impact, would need per-route typing decisions).
- "Must not mix styles per endpoint" is enforced only for literal returns;
  dynamic bare returns via helpers remain statically invisible.

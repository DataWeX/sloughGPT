---
id: 20261004_tokens_router_nonetype_deref
title: "tokens router NoneType deref (runtime: 'NoneType' object is not subscriptable on balance/usage/topup paths)"
status: done
date: 2026-10-05
tags: [api, bug, tokens, auth]
board_id: f01852a3-ce8d-4e2c-955e-6393979f8070
board: f01852a3-ce8d-4e2c-955e-6393979f8070
hash: aa0826ad95f8ffd3fed5cf79383cba3590321b3a8a6fd158172bf04f3857bb0e
---

## Original evidence (2026-10-04)

Verification sweep 2026-10-04 on worktree `/home/mana/Documents/sloughgpt-api-std @ 5bfbb8e08`
(full suite 619F/45211P/180E in 1h56m; logs in `~/.cache/sweep-logs/`).

`test_tokens_router.py` 12F — 9 fail with
`domain.infrastructure._internal.errors.AppError: 'NoneType' object is not subscriptable`.
Six router methods catch an exception and log it (`slo.routers.tokens:tokens.py` lines 54, 65, 81,
93, 114, 136).

## ROOT CAUSE — the card's theory was WRONG

The card guessed *"tokens store or provider not initialized (None)"*. **Not the store.**
Reproduced on this branch and the traceback shows the None is **`auth_user`**:

```
account = service.get_balance(auth_user["id"])
E   TypeError: 'NoneType' object is not subscriptable
apps/api/server/routers/tokens.py:48
```

`require_auth_if_enabled` (`infrastructure/auth.py:347`) **returns `None` whenever
`SLO_AUTH_REQUIRED` is false — the DEFAULT** — and never reads the `X-User-Id` header the test
was sending. `tokens.py` indexed `auth_user["id"]` unconditionally at 6 sites.

**So this was a real production bug, not a test artifact: with auth off (the default), EVERY
`/tokens/*` request 500'd and logged an exception. Token billing was dead in the default config.**

`tokens.py` was the ONLY router doing a bare `auth_user["id"]`; the rest already guard
(`jobs_api.py:113 if auth_user and ...`, `errors.py:277 (auth_user or {}).get(...) or "anonymous"`).

### The fix

`infrastructure/auth.py:334` already had the purpose-built helper — **`audit_user(auth_user)`**,
docstring: *"payload dict from `require_auth_if_enabled`, or **None when auth is disabled** …
otherwise `anonymous`"* — but it had **ZERO call sites**. Wired it into all 6 sites, plus
`auth_user: dict | None` annotations (it could always be None).

## Test repairs (`packages/core-py/tests/test_tokens_router.py`)

The test was doubly wrong beyond the router:

1. **Flat vs envelope.** Asserted `data["balance"]`, `"canAfford" in data`. `success_response`
   returns `{"status": "success", "data": …}`. All 8 sites now read `response.json()["data"]`.
2. **400 vs 422.** 3 sites asserted 400; `TopUpRequest.amount` is `ge=1, le=1_000_000` and
   `UpgradeRequest.tier` has a `pattern=`, so pydantic rejects **before** the router → 422.
   Confirmed production returns 422 too: `_request_validation_error_handler` returns
   `HTTP_422_UNPROCESSABLE_CONTENT`, and `main.py:343` registers `register_all_handlers`.

### Order-dependence I almost shipped (and fixed)

Mapping everyone to `anonymous` made all 12 tests **share one billing account**. Proven
order-dependent — running topup first made `test_get_balance_creates_account` fail
(`600 != 500`). Would have broken under `-k`, shuffling, or any reordering.

Fixed by giving each test a **private account**: a function-scoped `app` fixture overrides
`require_auth_if_enabled` with a `uuid4`-derived identity. Reversed-order run now passes.
Added `TestAuthDisabled::test_balance_falls_back_to_anonymous` — the regression test for the
actual bug (clears the override → real dependency → `userId == "anonymous"`, 200 not 500).

## VERIFICATION

Committed `90a1c1b95`. 

- `test_tokens_router.py`: **12F → 13 passed** (12 + new regression test)
- reversed-order subset: passes (was failing)
- **no logged `slo.routers.tokens` exceptions** during the run (acceptance item 1)
- orphaned unit copy `apps/api/server/tests/test_tokens_router.py`: 11 passed
- service layer `test_token_service.py` + `test_billing_token_service.py`: 103 passed
- `ruff check` + `ruff format --check`: clean

## MAJOR DISCOVERY — 97 orphaned test files

`packages/core-py/pytest.ini` is not the issue; the root `pytest.ini` says:

```
testpaths = packages/core-py/tests tests apps/cli/tests
```

**`apps/api/server/tests/` is in NO testpath → 97 test files are collected by NOTHING (0 items).**

Consequences:
- The "passing canonical" `apps/api/server/tests/test_tokens_router.py` is **dead code**.
- The ONLY tokens test that ever ran was the broken one in `packages/core-py/tests/`.
- Two copies are NOT duplicates: the running one uses the **real** billing service
  (integration), the orphaned one mocks it (unit). Both retained.

**→ needs its own card: decide the home (`tests/server/` is the real running home, 61 files)
and either add `apps/api/server/tests` to testpaths or migrate+delete.** Related to
`33773fce-b` (refactor-drift backlog) and the 86 duplicate test basenames in card `d2803a3c-5`.

## COORDINATION

Response-envelope contract pinned by card behind `254615744` (API.md truth-up) — the envelope
`{"status": "success", "data": …}` is what the repaired tests assert.

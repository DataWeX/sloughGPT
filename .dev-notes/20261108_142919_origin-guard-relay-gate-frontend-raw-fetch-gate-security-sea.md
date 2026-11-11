---
id: 20261108_142919_origin-guard-relay-gate-frontend-raw-fetch-gate-security-sea
status: done
board: 20261108_142919_origin-guard-relay-gate-frontend-raw-fetch-gate-security-sea
title: "Origin guard + relay gate + frontend raw-fetch gate (security seam)"
tags: [security, origin-guard, ssrf, frontend, http-client]
sprint: ""
gh: null
---

# Origin guard + relay gate + frontend raw-fetch gate (security seam)

Session 2026-11-10 (origin-guard lane, built directly after Gen hit the free-model limit; batch C→A→B approved by user 2026-11-09).

## Premise correction (at expert review)

No browser→provider key path exists to gate — the provider key lives in backend settings. The real exposure is **who may call**: `SLO_AUTH_REQUIRED` defaults off (anonymous on every endpoint), and CORS only blocks *reading* a response, not *sending* it — so a cross-origin SIMPLE POST (form/text-plain, no preflight) can execute state-changing actions today.

## What landed (branch `fix/origin-guard-relay-gate`, from origin/main `0175b4a87`)

1. **Origin guard** — `apps/api/server/infrastructure/origin_guard.py`: `OriginGuardMiddleware` registered in `get_configured_middleware` between `CorrelationIdMiddleware` (outer, so the 403 envelope carries the correlation ID) and `ReadinessGateMiddleware`. Unsafe methods (POST/PUT/PATCH/DELETE) with a foreign `Origin` → `403 E_BAD_ORIGIN` (hand-built envelope via `error_response` — raising AppError inside BaseHTTPMiddleware surfaces as 500). Allow rules mirror main.py's CORS block (untouched): loopback regex any-port + `SLO_CORS_ORIGINS` env + same-origin netloc==Host. No-Origin (curl) passes; `/v1`+`/v1/...` path-exempt; OPTIONS never in the unsafe set (CORSMiddleware answers preflight outermost). Exported `require_first_party_origin` dependency = per-route projection.
2. **Webhook SSRF** — `validate_url_not_private(url)` at `WebhookStore.register()`, the single seam where caller-chosen URLs enter the system (registration-time only; delivery-time TOCTOU/DNS rebinding explicitly out of scope).
3. **Planner relay gate** — `proxyRequest` refuses calls without `x-slo-relay: 1` (403-before-fetch), pins resolved host to `new URL(PLANNER_API_BASE).host` (closes `@evil.com` userinfo escapes), strips the relay header before forwarding; sole call site `transition/route.ts` passes it.
4. **Frontend fetch gate** — `lib/fetch-gate.test.ts`: zero-baseline source scan for raw `fetch()` outside the http-client facade; 11-entry allowlist shrink-only + honesty test (an entry that stops carrying a raw fetch fails the gate). Scan found exactly one violation — `lib/dev-log.ts:149` — migrated to `apiPost('/errors/logs/ingest', …, {keepalive, silent, skipCircuitBreaker})`; `keepalive` plumbed through `RequestOptions`/`request.ts`; `skipCircuitBreaker` zeroes retries (old raw fetch was retry-free); `LogTransport.flush()` became async (apiPost crosses microtasks).

## Pre-existing bug found + fixed en route

`ssrf.py::validate_url_not_private` swallowed its own rejection: `"E_BAD_REQUEST" in str(e)` can never match `AppError.__str__` (message only) — the check was a **silent no-op for all callers** (webhooks + `routers/datasets.py` ×3). Fixed with `except AppError: raise`; blast radius verified (no test depended on the swallow); datasets' SSRF rejection now actually rejects (intended).

## Fixtures

NXDOMAIN hosts (`a.example`, `b.example`, `hooks.example*`) → `example.com` variants in 3 test files (2/1/5 sites) — DNS-at-test-time precedent (datasets' github.com tests); emit/delivery tests mock `notify_event`, so no live fetches.

## Evidence (gates)

- **Python RED watched** (14 failed) → closure **208 passed**: `test_origin_guard.py` 20 + `test_webhook_url_guard.py` 6 + descriptors + store + consolidation + middleware + errors_router + llm_v1 + contracts; `ruff check`+`format` clean (9 files); `gen_router_manifest --check` 0 (43 modules).
- **TS RED watched** (planner-proxy gate 4 + fetch-gate zero-baseline 1) → `planner-proxy` 9 + `fetch-gate` 6 + `dev-log` 34 + transition `route` 2 + `http-client` 70 + `planner-client` = **145 passed**; `npm run test:changed` **383 files / 4472 tests green** (one fixture retargeted: `route.test.ts` expects the relay header); `tsc --noEmit` 0; eslint 0 errors, 741 warnings (< 900 budget).
- **Card's 7 required behavior tests all present**: foreign-Origin POST→403; curl/no-Origin→200; loopback Origin→200; /v1 exempt; relay-without-header→403 (no fetch); webhook `169.254.169.254`→400; raw fetch outside facade→gate failure (synthetic-violation test proves the gate fails).
- **Must-not-affect honored**: /v1 (exempt tests), SSE/WebSocket (GET-only; BaseHTTPMiddleware never sees WS), api-explorer + worker asset fetches (allowlisted with reasons), loopback CORS rule + preflight amortization (main.py untouched), Zama/3D untouched.
- **Push**: `e9973ce4e` (implementation) + `44feab011` (AGENT_SYNC) → `origin/fix/origin-guard-relay-gate`, 0-ahead. Lane branch only — **not merged to main** (no user sanction).

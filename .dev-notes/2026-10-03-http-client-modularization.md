---
id: 20261003_http_client_modularization
title: "http-client.ts modularization: 1443-line stacked/disjointed core — split + unify request paths (investigation + proposal, user reviews design first)"
status: doing
date: 2026-10-03
tags: [web, frontend, architecture, refactor, tech-debt]
board_id: 656023ab-84a8-477b-9eb3-b677a05f9fa3
board: 656023ab-84a8-477b-9eb3-b677a05f9fa3
hash: 44f1fc0ba15ac4c7fd361847f883b70d8ab7e950a7ee43aa371d4b756853d936
---

FINDING (2026-10-03, user asked "how stacked and disjointed is http-client? maybe kanban it"): `apps/web/lib/http-client.ts` is 1443 lines carrying ~8 responsibilities in one file — error model (ApiError), correlation-id tracking, URL resolution (_resolveUrl + SAME_ORIGIN_RE carve-out for planner/calendar L262-267), InterceptorManager, HttpCache, CircuitBreaker, Throttler, request dedup (`_inflight` L524 + `_recentDedup` L525 TTL map), a retry engine with TWO policies (generic RETRYABLE_STATUSES {408,429,502,503,504} L174 vs DOCSTORE_RETRYABLE_STATUSES {408,429,500,502,503,504} + DOCSTORE_MAX_RETRIES=4 / BASE_DELAY=300 L204-206, selected by URL sniffing `_isDocstoreUrl` L234), long-op path (`_isLongOperationUrl` L253, LONG_OPERATION_TIMEOUT_MS=120_000 L242), a ~350-line `request()` core (L542-890), 5 verb wrappers (apiGet L894 / apiPost L990 / apiPut L1002 / apiDelete L1014 / apiPatch L1026), and streaming.

MEASURED 2026-10-03 (grep token occurrences across apps/web *.ts/*.tsx, tests included): apiGet 878, apiPost 721, apiDelete 173, apiPatch 94, apiPut 89, apiClient 653, streamSSE 50, httpClient 22, authFetch 22, createHttpClient 13, bare `request(` 7 — 115 files import from http-client, and **54 raw `fetch(` call sites exist outside http-client.ts** (direct Core-Infrastructure-Sync-Rule violations: raw fetch is banned). Test anchor: `apps/web/lib/http-client.test.ts` = 933 lines / 74 tests (must stay green).

DISJOINTED: ~6 parallel request surfaces — bare apiGet/apiPost/apiPut/apiDelete/apiPatch + request(), `apiClient` singleton (L1096), `httpClient` singleton (L1274), `createHttpClient()` factory (L1277), `authFetch` (L1298), `streamSSE` (L1328) which duplicates URL-prefixing + error formatting instead of going through _resolveUrl/request. 14 special-case branches (docstore/same-origin/long-op/dedup/stream). Every new feature adds another conditional to the same core.

WHY NOW: root AGENTS.md gained "Endpoint & Transport Rule (core-first, descriptor-projected)" (AGENTS.md:79) — future capability projections (generated routes, SSE frames, typed client helpers) must land in http-client as THE single source of truth (Core Infrastructure Sync Rule). Stacking them onto the current monolith deepens the problem; the transport rule needs a modular host to land in.

TASK (investigation + design proposal ONLY — user reviews before any refactor, card E pattern):
1. map current responsibilities + all 6 request paths and their consumers (grep apiGet/httpClient/authFetch/streamSSE usage counts — seed counts above, verify and attribute per path);
2. propose a module split (e.g. lib/http/: core request engine, retry policies, url resolution, cache/circuit/throttle, streaming, auth) preserving the PUBLIC surface — every existing export keeps working, zero consumer changes in phase 1;
3. propose unifying the 6 paths behind the factories (streamSSE and authFetch adopting _resolveUrl/request error model);
4. list risks (interceptor order, dedup keys, docstore policy isolation) and a test strategy (existing vitest suite must stay green).

ACCEPTANCE: proposal doc presented to user and approved before code; no behavior change in phase 1; http-client remains THE single source (AGENTS Core Infrastructure Sync Rule); shared-workspace (root node_modules only, zero installs); no new docs files (proposal lives in the card/note until approved).

COORDINATION (grep of .kanban/board.jsonl 2026-10-03, READ-ONLY — none of these cards modified): context = root AGENTS.md "Endpoint & Transport Rule" section (edited this turn by the main session — reference only, leave AGENTS.md alone). Existing http-client/refactor cards found and referenced:
- `e6688c08-2fa6-4898-b31c-8229745d1ee1` [done] "Frontend stops re-firing inference: no retry amplification, no immortal startup polls" (tags frontend/http-client/reliability/performance) — owns retry-amplification semantics; any retry-policy split must not regress it.
- `0e1fa738-173d-40bc-a1c4-ee398d870463` [done] "Fix runtime errors: data-root off-by-one, silent 400s, planner 404s, sessions 405, VM auto-save, Tailwind class" (tags docstore/http-client) — origin of the same-origin planner/calendar carve-out + the 400-no-retry rule.
- `20260923_017_sse-abort-no-error-log` [done] "SSE streamSSE: don't error-log/forward intentional aborts" (tags http-client/adopts-http-client) and `20260809_161949_vm-console-persist-train-config-reset-fix-streamsse-regressi` [done] — streamSSE behavior contract.
- `e67e637b-095e-4f74-8bf1-7733ad693e0a` [done] "chat router owns POST /chat + /chat/stream" (streamsse) — server side of the SSE path.
- `6d0ebf37-6287-45b8-8bce-c1f1b8a36798` [done] Site Doctor surfacing — representative consumer (apiGet/apiPost only).
- `56b49cf1-72db-48e2-9dde-d857ab214f10` [todo] "Triage pre-existing test-vs-impl drift baseline" (tech-debt) — separate triage lane, do not fold in.
- `20260924_054_utility_libraries_timestamps` [done] "Build utility libraries ... migrate all callers" (refactor) — PRECEDENT to copy: split into modules + keep a facade, public surface unchanged.
- in_progress http-client-tagged consumers only (no overlap with lib/http-client.ts): `7f879f56-...` (chargectl), `a92c4a31-...` (workspace search).

SHARED WORKSPACE — index updated: N/A (no new app/package/doc registered; proposal stays in this card/note until approved, docs/INDEX.md untouched); shared copy reused: root node_modules + existing vitest only, ZERO installs; coordination named: the board cards listed above are referenced, not modified; AGENTS.md Endpoint & Transport Rule is context only.

---

## PROPOSAL (2026-10-04 — investigation complete, AWAITING USER APPROVAL before any code)

### Measured state (re-verified 2026-10-04, file-level counts excluding http-client.ts itself)
- 1443 lines, 24 export symbols; importers = 115 files (62 `@/lib/http-client`, 53 `./http-client`, incl. one `* as http` namespace import) — both specifiers resolve to the ONE file, so a facade file covers everything.
- Path weights (files): verbs apiGet 173 / apiPost 139 / apiDelete 59 / apiPut 27 / apiPatch 26 · apiClient 32 · streamSSE 14 · authFetch 9 · bare request( ~7 · httpClient 1 (near-dead) · createHttpClient 1.
- NEW: a THIRD retry policy lives inside streamSSE (own attempt/delay loop) alongside GENERIC {408,429,502,503,504}x2/500ms and DOCSTORE {...,500...}x4/300ms.
- NEW: split-brain placement — HttpCache + Throttler are applied in the verb WRAPPERS (apiGet L905-932; apiPost/Put/Delete/Patch L992+ acquire/release) while CircuitBreaker is recorded inside the request() core (L757/782/835) — resilience trio not co-located.
- authFetch = most divergent path: raw fetch(), NO _resolveUrl (misses SAME_ORIGIN_RE planner/calendar carve-out), no corr-id, no retry, no ApiError model.
- streamSSE does use _resolveUrl + corr-id but bypasses request() entirely (own headers/retry/logging; abort≠error contract owned by card 20260923_017 — must be preserved).

### Phase 1 — split, ZERO behavior change (public surface byte-preserved)
Create `apps/web/lib/http/`, one-way dependency flow:
`types.ts` ← `errors.ts` (ApiError) ← `url.ts` (_resolveUrl, _isDocstoreUrl, _isLongOperationUrl, SAME_ORIGIN_RE, LONG_OPERATION_PATHS) ← `retry.ts` (both policies + _isIdempotent + selector) ← `corr-id.ts` ← `interceptors.ts` (InterceptorManager) ← `resilience.ts` (HttpCache/CircuitBreaker/Throttler + the three `_global*` singletons L877-880) ← `request.ts` (core engine L542-890) ← `verbs.ts` (5 wrappers + _doGet; cache+throttle wiring) ← `api-client.ts` (createApiClient/apiClient) ← `client.ts` (_createHttpClient/httpClient/createHttpClient) ← `auth.ts` (authFetch) ← `sse.ts` (streamSSE).
`lib/http-client.ts` becomes a pure re-export facade (`export * from './http/index'`) — zero consumer edits, both specifiers + namespace import intact.
GATE: existing 74 tests (933 lines) UNTOUCHED and green; tsc 0; eslint delta 0.

### Phase 2 — unify the 6 paths (each step behind its own green gate)
1. authFetch → wrapper over `request({raw:true})`: gains _resolveUrl + corr-id + ApiError (9 files); add scoped-route test.
2. streamSSE → shares header assembly/corr-id/retry-policy object with core; policies become a named map GENERIC|DOCSTORE|SSE; keep abort-logging contract (describe('streamSSE') guards).
3. httpClient singleton (1 consumer) → migrate that one file, then alias/delete (one fewer client).
4. apiClient + factory → delegate to the same request() with per-client defaults (configure() maps to RequestOptions) — one engine, per-client config.
5. Raw-fetch ratchet: lint rule banning NEW bare fetch( outside lib/http/ (existing ~32 files grandfathered, separate lane).

### Risks
- **Module-instance duplication** (top risk): the facade must RE-export, never re-implement — otherwise interceptors, `_inflight`/`_recentDedup` and `_global*` fork into two instances and dedup/order break silently. Single import graph enforced.
- Dependency cycles inside lib/http/ (url/retry must never import request).
- Docstore policy isolation (URL-sniff selector; describe('docstore 400 handling') guards).
- Retry-amplification contract (card e6688c08 done) + non-idempotent safety (describe('retry safety…')) must stay green.
- Interceptor execution order in request() byte-identical in phase 1.

### Test strategy
Phase 1: zero test edits — 74 green proves no behavior change. Phase 2: +authFetch URL-resolution, +streamSSE policy-sharing, +1 consumer-migration test; then split http-client.test.ts per module (copy utility-libraries precedent card 20260924_054: split modules + keep facade).

### Out of scope
Transport-rule projections (typed ToolSpec helpers) = phase 3, separate card after this lands. Raw-fetch backlog cleanup = separate lane. Counts note: 2026-10-03 note counts are OCCURRENCES (apiGet 878), today's are FILES (apiGet 173) — same universe, different units.

---

## PHASE 1 LANDED (2026-10-04)

- Branch `feat/http-client-split` commit `dd29a1305` → merged to main as `44237ae9b` (worktree `/home/mana/Documents/sloughgpt-httpc`).
- 1395-line `apps/web/lib/http-client.ts` → 33-line pure facade + 15 modules in `apps/web/lib/http/` (errors, types, corr-id, url, retry, interceptors, resilience, state, request, verbs, api-client, client, auth, sse, index) with one-way dependency flow.
- Gates (all green): export surface diffed 33/33 vs HEAD · http-client.test.ts untouched, 70/70 · full lib suite 1515/1515 (116 files, 44s) · tsc 0 errors · eslint 0 problems.
- Design principle applied per user: "as modular and generic as possible" — one module per concern; single-instance state (`_global*`, `_inflight`/`_recentDedup`, `httpDefaults`) each owned by exactly one module; facade re-exports, never re-implements (guard against instance duplication).
- Pre-existing lint warnings silenced while moving (behavior-identical): unused `revalidation`/`revalidationPromise` assignments dropped from their promise chains.
- Phase 2 (authFetch → request({raw}), streamSSE retry-policy sharing, httpClient-singleton retirement, raw-fetch ratchet) remains proposal-only — awaiting user go-ahead.

---

## LAYOUT MOVE LANDED (2026-10-04) — flat apps/lib/

- User directive "move to apps/lib" + "remove that http [folder]": the 15 modules now live **flat in `apps/lib/*.ts`** (no `http/` subfolder), commit `d5244478d`, merged to main as `91a216808`.
- Facade `apps/web/lib/http-client.ts` → `export * from '../../lib/index'`; consumer import paths unchanged.
- Intra-module `./x` imports untouched (siblings); web-lib deps re-based to `../web/lib/{config,auth,dev-log,sse-client,error-store,api-monitor-store}` (one level up — flat dir, not http/).
- Registered `apps/lib` in `apps/README.md` (same change, shared-index rule); added `apps/lib/.eslintrc.json` → `@sloughgpt/eslint-config/next`.
- Gates re-run green after move: surface 33/33 vs pre-split `c9d7c94cf` · tsc 0 · eslint rc=0 · full lib suite 1515/1515.

---
id: 20261108_134211_runtime-config-descriptor-for-web-client-flagslimitsendpoint
status: wip
board: 20261108_134211_runtime-config-descriptor-for-web-client-flagslimitsendpoint
title: "Runtime config descriptor for web client (flags/limits/endpoints)"
tags: [api, frontend, decision, contract, feature-flags]
sprint: ""
gh: null
---

# Runtime config descriptor for web client (flags/limits/endpoints)

User approval 2026-11-11: design "Go (Recommended)" — build as specified.

## Audit verdict (gate 0, completed 2026-11-11)

One genuine gap: **server feature flags invisible to the browser** — `domain/shared/_internal/feature_flags.py` holds 16+ runtime flags (status/version/since, `SLO_FF_*` env overrides, append-only `config/flag_history.jsonl` ledger) read across soul/managers/train/consciousness/quant/response-cache, with **zero** exposure (`grep feature_flags apps/api/server/routers/*.py` = none; manifest has only the deprecated `config` module).

Everything else the card lists is covered or correctly build-time:
- user prefs + provider endpoints (key redacted) → existing `GET /settings/*`
- `routers/config.py` → deprecated (grandfathered, untouched)
- `PUBLIC_API_URL` → **stays compiled** (chicken-egg: a known origin is required to fetch the doc that would carry origins); `NODE_ENV`/`DEV` correctly build-time; `PLANNER_API_BASE` server-route concern
- UI-local `MAX_*` caps → local presentation buffers, not server truth
- no server rate/upload limits exist for the UI to show

## Design (approved)

- ONE descriptor `runtime_config.get` → `GET /runtime-config`, idempotent → GET, `auth_scope="authenticated"` (card said "public"; refined — nothing needs pre-auth access), via `ToolSpec` + `create_router` (endpoint rule: descriptor is the contract, route is a projection).
- Payload `{config_version, flags: [{name, status, enabled, version, since, description}]}` sorted by name; `enabled` = **resolved** (env override wins — what a consumer acts on); no `env_var`, no settings, no keys, no secrets.
- `config_version` = sha256 over sorted `(name, status, version, enabled)` (cheap, 16 flags) = the `ETag`; `Cache-Control: private, max-age=30` (card's cache ladder); optional `?v=` sugar.
- Frontend consumer `apps/web/lib/runtime-config.ts`: `fetchRuntimeConfig()` via `apiGet` only (http-client rule) + `isFlagEnabled(name)` with 30s in-memory TTL matching the header. **No UI wiring, no page** (surface ladder) — a real consumer gets wired when a journey needs one.
- VERIFY (card): contract test pinning descriptor shape + env-override resolution + ETag/Cache-Control; web test proving flag reads come from the response.

## Work

Branch `feat/runtime-config`, worktree `fe08f8/runtime-config`, base `origin/main` `68e4f2e97`. Testmon DB seeded via sqlite backup API from the live session-contract DB; node_modules hardlink-copied (disk 97%).

## Not touched

`routers/config.py`, `settings.py`, `PUBLIC_API_URL`, any UI/page, `domain/` (baseline unaffected), secrets/provider keys, model-facing tool registry (tier boundary — this is a site capability only).

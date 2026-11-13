---
id: card090
title: "Card 090 — Provider visibility: live registered/text_provider on GET + chat indicator"
status: done
priority: medium
created: 2026-09-29
source: card 087 investigation
---

## Ask

Gaps from card 087: (1) `GET /settings/providers/api` returns only
`_provider_public()` (no `registered`, no `text_provider`) — PATCH returns
them, so the settings status chip is blank on page load and after a server
restart even though `startup.py` re-applies the provider. (2) The chat UI
never shows which provider answers (local vs external API).

## Done when

- GET returns `registered` + `text_provider` (same computation as PATCH /
  `configure_api_provider` outcome); settings chip live on load.
- Chat UI shows the active text provider (from model status
  `text_provider`, already served by `models.py:1777`).
- Backend `test_settings_provider_router.py` + FE card/controller tests green;
  ruff / tsc / eslint clean.

## Evidence

Landed 2026-09-29 on `docs/api-doc-gap-audit` branch/worktree `.wt-audit`
(uncommitted; disjoint file sets from cards 083/088 — split at commit time).

**Backend** (`apps/api/server/routers/settings.py`):
- `_provider_live_state()` — public facade only (`domain.models.get_provider`
  + duck-typed `_text_name`, same pattern as `models.debug_providers`);
  initially written with `_internal` reach-ins, caught by the facade guard
  test `test_internal_reachins_do_not_grow` and rewritten (baseline restored).
- GET now returns `registered` + `text_provider` → settings chip accurate on
  page load and after restart (startup applies provider silently).
- PATCH response now includes `text_provider` for parity.
- Tests: +3 (`live state defaults`, `reflects registered provider after
  restart` via fresh client, `patch returns registered + text_provider`).

**Frontend**:
- `ProviderApiSettings`: `registered?: boolean | null`, `text_provider?: string | null`.
- New `ChatProviderChip` in the chat toolbar right cluster — renders only
  when `text_provider === 'api'` (local-model chat stays uncluttered),
  30s poll + mount fetch, `sm:`+ only, title tooltip shows model + endpoint.
- Card test: `Registered` chip now asserted from GET on load (no save needed).

**Pre-existing flake fixed en route** (`lib/error-store.ts`):
settings/page.test.tsx is the only page test not mocking http-client → real
fetch retries fire http-client's fire-and-forget `addError` after vitest
teardown, where the mocked extractor yielded `undefined` →
`.toLowerCase()` TypeError → full-suite `Errors 1` (exit 1) intermittently.
Hardened `getSeverity` + `addError` message with `?? ''`. Also created the
missing shared `node_modules` symlinks for `.wt-audit` (same lesson as
card 080; `.git/info/exclude` covers them).

## Gates

- server suite: **1209 passed / 0 failed** (includes facade guard green)
- web full suite: **837 files / 8037 tests passed, 0 unhandled errors,
  exit 0** (first fully-green full run this session; was `Errors 1` exit 1)
- `tsc --noEmit` exit 0; eslint touched files exit 0 (warnings pre-existing)
- targeted vitest (chip/toolbar/card/controller): 46 passed

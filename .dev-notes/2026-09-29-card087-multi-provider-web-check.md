---
id: 20260929_087
board: 7e2af9fe-53a1-4216-85fd-b5b514b2a42c
title: "Card 087 — Check: multi-provider web UI + website optimization status"
status: done
date: 2026-09-29
tags: [web, check]
---

# Check: multi-provider web UI + website optimization status

## Background

Asks 2026-08-28: 'build our applications so we can use our web ui with another api of ai... like openai' and 'we still need our website optimized well, use qa agent'. Verify what exists (api-keys page, provider config), report gaps -> concrete cards. Investigation only until confirmed.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Findings (2026-09-29, investigation only — no code changes)

### 1. Multi-provider web UI: DONE (merged 2026-09-22)

Card `2026-09-22-openrouter-provider-wiring` (status done) delivered the full
chain — verified end-to-end in code:

| layer | artifact |
|---|---|
| UI | `SettingsExternalProviderCard` on settings page (endpoint URL, API key, model id, enable toggle) + `registered` status chip + save toast |
| client | `settings-controller.getProviderApi/updateProviderApi` via http-client (`GET/PATCH /settings/providers/api`) |
| API | settings router GET (never leaks key: `api_key_set` only) + PATCH applies at runtime |
| runtime | `configure_api_provider()` registers `ApiProvider` under `"api"`, points `ProviderRouter` text provider at it; disable → falls back `native-c` → `slonet-native`; processor pipeline (personality/style/tools) intact |
| startup | `startup.py:1483` re-applies persisted provider settings on boot |
| embeddings | separately generic since card 079 (`embedding.api_key`/`base_url`) |
| api-keys | pages `/(app)/api-keys` + `/workspace/settings/api-keys`, 3 cards with tests; backend `/security/keys` CRUD |

Key handling correct: server-stored, raw key never returned, PATCH-only
update, distinct from service URL + HF token (per card-062 constraint).

### 2. Website optimization: PARTIAL — QA pass never ran

Exists: `next.config.js` tuned (standalone prod, AVIF/WebP + device sizes,
lucide `modularizeImports`, v86 external/stub, `transpilePackages` strui,
TS errors fail build); 10 `next/dynamic` lazy routes; Rust gateway owns
compression (card 073 edge-cascade); vitest + tsc + eslint gates green
(8031 web tests, card 080).

Missing: **zero** Lighthouse/Web Vitals/bundle-analysis history in notes or
docs — the 'use qa agent' half of the ask never happened; no bundle analyzer
script; recharts pinned un-modularizable (v2 carve-out in next.config) so
chart pages pull the full library; no build artifacts present to measure
(dist absent, stack not running).

### Gaps → cards

- **card 090** — provider visibility: GET `/settings/providers/api` omits
  `registered`/`text_provider` (only PATCH returns them) so the settings chip
  is blank on page load / after restart despite startup applying the
  provider; chat UI never shows which provider answers.
- **card 091** — website optimization QA: run Lighthouse + bundle analysis
  against a prod build ('use qa agent'), fix top findings.

---
title: Mole becomes a plugin, not a page
date: 2026-10-28
tags: [mole, plugins, frontend, backend, cleanup]
status: done
---

# Mole becomes a plugin, not a page

## Ask
User: keep the Mole UI quality ("nice UI") but stop shipping Mole as a
hardwired page + nav pointer — it should be a plugin, loaded and toggled
through the EXISTING plugin stack (do not invent a second framework).

## Existing stack (reuse, per AGENTS.md "check before building")
- `domain/plugins/_internal/plugins.py` — PluginManager, hooks,
  `load_from_directory()` (expects module-level `plugin_class`, loads from
  repo-root `plugins/`).
- `apps/api/server/routers/plugins.py` — GET /plugins, enable/disable,
  POST /plugins/reload (scans repo-root `plugins/`).
- `apps/web/lib/plugins-cloud-controller.ts` — typed client (http-client).
- `apps/web/app/(app)/plugins-cloud/page.tsx` — Plugins tab (list + badge).

## Work
1. `plugins/mole.py` — MolePlugin (plugin_class) w/ metadata + hooks.
2. Frontend: Mole plugin panel on the Plugins tab (report UI reused from
   `components/mole/*`, gated on plugin enabled).
3. Delete `app/(app)/mole/` page + test; remove nav entry (navigation.ts),
   `nav.mole` locale keys, route-contract.json row.
4. Redirect `/mole` -> `/plugins-cloud` (redirect count 46 -> 47; proxy pin).
5. page-baseline.json: drop mole row, GRANDFATHERED_CEILING 73 -> 72.
6. Backend test: plugin loads from directory, appears in GET /plugins.

## Gates
pytest (plugins, mole router), vitest (route-contract, page-journey-gate,
redirect-plugin, proxy, plugins-cloud), tsc, eslint.

## Landed
PR #226 squash-merged as `260df315e`; sync entry `0c738d756`; board card
moved to done; branches deleted both sides. Full AGENT_SYNC narrative:
docs/AGENT_SYNC.md entry dated 2026-10-28 (mole-plugin lane).

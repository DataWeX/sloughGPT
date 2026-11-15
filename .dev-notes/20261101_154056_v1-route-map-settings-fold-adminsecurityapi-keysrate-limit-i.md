---
id: 20261101_154056_v1-route-map-settings-fold-adminsecurityapi-keysrate-limit-i
status: done
board: 20261101_154056_v1-route-map-settings-fold-adminsecurityapi-keysrate-limit-i
title: "v1 route map /settings: fold admin/security/api-keys/rate-limit into /settings — merge security.py + api_keys.py"
tags: ["roadmap", "v1", "frontend", "routers"]
sprint: ""
gh: null
---

# v1 route map /settings: fold admin/security/api-keys/rate-limit into /settings — merge security.py + api_keys.py

Parent: 20261101_140445_v1-route-map-land-the-7-pending-targets-models-souls-knowled (umbrella; cards cannot be updated via CLI so linkage lives here).
Route Map row /settings (was pending; verified 2026-11-01): pages on disk = app/(app)/settings/page.tsx + app/(app)/settings/workspace/page.tsx — /admin, /security, /api-keys, /rate-limit pages already removed (stale table). workspace/settings/* pages are the workspace persona area — do NOT absorb them into the Alex journey without a citation. Routers: apps/api/server/routers/settings.py + security.py, plus routers/api_keys.py which the PRODUCT_ENGINEERING Dead Code row says: merge into security.py, delete api_keys.py.
Work: journey-cited merge of security + api_keys into the settings surface, redirect old paths, shrink page-baseline.json.
Journey citation required at pickup: a ux/ flow in docs/UX_FLOWS.md (Page Rule test 1); no citation = bounce back before code.
Gates at pickup: vitest run of apps/web/lib/page-journey-gate.test.ts green; tsc exit 0; eslint 0 errors; page-baseline ratchet (any new page.tsx must be catalogued in apps/web/lib/page-baseline.json citing a ux/ journey; grandfathered/unmapped counts may only shrink; deleting a page = delete its entry + decrement GRANDFATHERED_CEILING); change-scoped tests via scripts/run_affected.sh; tests/test_layer_conformance.py green.
Surface ladder: cheapest surface that works — global banner < inline widget < chat mode (/chat?mode=) < plugin panel < operator page < journey page; a page is the most expensive rung. Do not add a page.tsx unless the journey citation survives the ladder.
SEQUENCING: this card touches apps/web/lib/navigation.ts + apps/web/lib/page-baseline.json, shared with every other v1 route-map card — must run SEQUENTIAL, never in parallel with sibling route-map cards (Gen batches must sequence them).
Router merges are descriptor projections (Endpoint Rule): no hand-built feature routers; merged routes delegate to domain.<pkg> public facades only — 0 _internal imports (tests/test_layer_conformance.py enforces).

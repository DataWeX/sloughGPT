---
id: 20261101_154058_v1-route-map-monitoring-merge-errorspy-sessionpy-routers-int
status: done
board: 20261101_154058_v1-route-map-monitoring-merge-errorspy-sessionpy-routers-int
title: "v1 route map /monitoring: merge errors.py + session.py routers into /monitoring"
tags: ["roadmap", "v1", "frontend", "routers"]
sprint: ""
gh: null
---

# v1 route map /monitoring: merge errors.py + session.py routers into /monitoring

Parent: 20261101_140445_v1-route-map-land-the-7-pending-targets-models-souls-knowled (umbrella; cards cannot be updated via CLI so linkage lives here).
Route Map row /monitoring (was pending; verified 2026-11-01): app/(app)/monitoring/page.tsx remains — /errors + /session pages already removed (stale table). /monitoring in NAV_SECTIONS (shortcut 7). Routers: apps/api/server/routers/system.py + errors.py + session.py.
Work: merge errors.py + session.py into the monitoring/system surface per journey citation, fold error/session views into /monitoring, redirect old paths, shrink page-baseline.json.
Journey citation required at pickup: a ux/ flow in docs/UX_FLOWS.md (Page Rule test 1); no citation = bounce back before code.
Gates at pickup: vitest run of apps/web/lib/page-journey-gate.test.ts green; tsc exit 0; eslint 0 errors; page-baseline ratchet (any new page.tsx must be catalogued in apps/web/lib/page-baseline.json citing a ux/ journey; grandfathered/unmapped counts may only shrink; deleting a page = delete its entry + decrement GRANDFATHERED_CEILING); change-scoped tests via scripts/run_affected.sh; tests/test_layer_conformance.py green.
Surface ladder: cheapest surface that works — global banner < inline widget < chat mode (/chat?mode=) < plugin panel < operator page < journey page; a page is the most expensive rung. Do not add a page.tsx unless the journey citation survives the ladder.
SEQUENCING: this card touches apps/web/lib/navigation.ts + apps/web/lib/page-baseline.json, shared with every other v1 route-map card — must run SEQUENTIAL, never in parallel with sibling route-map cards (Gen batches must sequence them).
Router merges are descriptor projections (Endpoint Rule): no hand-built feature routers; merged routes delegate to domain.<pkg> public facades only — 0 _internal imports (tests/test_layer_conformance.py enforces).

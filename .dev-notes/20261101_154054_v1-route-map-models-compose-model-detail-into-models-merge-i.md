---
id: 20261101_154054_v1-route-map-models-compose-model-detail-into-models-merge-i
status: done
board: 20261101_154054_v1-route-map-models-compose-model-detail-into-models-merge-i
title: "v1 route map /models: compose model detail into /models — merge infer.py + registry.py routers"
tags: ["roadmap", "v1", "frontend", "routers"]
sprint: ""
gh: null
---

# v1 route map /models: compose model detail into /models — merge infer.py + registry.py routers

Parent: 20261101_140445_v1-route-map-land-the-7-pending-targets-models-souls-knowled (umbrella; cards cannot be updated via CLI so linkage lives here).
Route Map row /models (PRODUCT_ENGINEERING.md, was pending; verified 2026-11-01): pages on disk = app/(app)/models/page.tsx + app/(app)/model/[id]/page.tsx — STALE table discovery: /infer and /registry pages already removed. Routers still separate: apps/api/server/routers/models.py + infer.py + registry.py. /models has NO NAV_SECTIONS entry today.
Work: land the /models target — journey-cited decision on model/[id] composition vs merge, fold infer.py + registry.py into the models surface, add the nav entry, redirect old paths, shrink page-baseline.json.
NOTE: sibling card 20261101_144013_model-loading-infrastructure-must-land-before-v1-release-use claims model-loading scope — coordinate at pickup to avoid double-building.
Journey citation required at pickup: a ux/ flow in docs/UX_FLOWS.md (Page Rule test 1); no citation = bounce back before code.
Gates at pickup: vitest run of apps/web/lib/page-journey-gate.test.ts green; tsc exit 0; eslint 0 errors; page-baseline ratchet (any new page.tsx must be catalogued in apps/web/lib/page-baseline.json citing a ux/ journey; grandfathered/unmapped counts may only shrink; deleting a page = delete its entry + decrement GRANDFATHERED_CEILING); change-scoped tests via scripts/run_affected.sh; tests/test_layer_conformance.py green.
Surface ladder: cheapest surface that works — global banner < inline widget < chat mode (/chat?mode=) < plugin panel < operator page < journey page; a page is the most expensive rung. Do not add a page.tsx unless the journey citation survives the ladder.
SEQUENCING: this card touches apps/web/lib/navigation.ts + apps/web/lib/page-baseline.json, shared with every other v1 route-map card — must run SEQUENTIAL, never in parallel with sibling route-map cards (Gen batches must sequence them).
Router merges are descriptor projections (Endpoint Rule): no hand-built feature routers; merged routes delegate to domain.<pkg> public facades only — 0 _internal imports (tests/test_layer_conformance.py enforces).

---
id: 20261101_154058_v1-integration-files-merge-imagespy-router-into-filespy-fron
status: done
board: 20261101_154058_v1-integration-files-merge-imagespy-router-into-filespy-fron
title: "v1 integration /files: merge images.py router into files.py (frontend already done)"
tags: ["roadmap", "v1", "routers"]
sprint: ""
gh: null
---

# v1 integration /files: merge images.py router into files.py (frontend already done)

Parent: 20261101_140445_v1-route-map-land-the-7-pending-targets-models-souls-knowled (umbrella; cards cannot be updated via CLI so linkage lives here).
Route Map row /files (partial; verified 2026-11-01): frontend done — /images absorbed into /files with a 307 (card 7839a7c6); /images page gone from disk. Routers still pending: apps/api/server/routers/files.py + images.py both exist.
Work: merge images.py routes into files.py behind the files facade, delete images.py, keep the 307 redirect behaviour, keep http-client.ts consumers green (update it FIRST if response shapes change — Core Infrastructure Sync Rule).
Journey citation required at pickup: a ux/ flow in docs/UX_FLOWS.md (Page Rule test 1); no citation = bounce back before code.
Gates at pickup: vitest run of apps/web/lib/page-journey-gate.test.ts green; tsc exit 0; eslint 0 errors; page-baseline ratchet (any new page.tsx must be catalogued in apps/web/lib/page-baseline.json citing a ux/ journey; grandfathered/unmapped counts may only shrink; deleting a page = delete its entry + decrement GRANDFATHERED_CEILING); change-scoped tests via scripts/run_affected.sh; tests/test_layer_conformance.py green.
Surface ladder: cheapest surface that works — global banner < inline widget < chat mode (/chat?mode=) < plugin panel < operator page < journey page; a page is the most expensive rung. This card is a router/merge task: do NOT add a page.tsx unless the journey citation survives the ladder.

SEQUENCING: this card touches apps/web/lib/navigation.ts + apps/web/lib/page-baseline.json, shared with every other v1 route-map card — must run SEQUENTIAL, never in parallel with sibling route-map cards (Gen batches must sequence them).
Router merges are descriptor projections (Endpoint Rule): no hand-built feature routers; merged routes delegate to domain.<pkg> public facades only — 0 _internal imports (tests/test_layer_conformance.py enforces).

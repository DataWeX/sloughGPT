---
id: 20261101_154058_v1-integration-training-consolidate-self-trainpy-lora-evalpy
status: done
board: 20261101_154058_v1-integration-training-consolidate-self-trainpy-lora-evalpy
title: "v1 integration /training: consolidate self_train.py + lora_eval.py + cloud_training.py into the training router (sidebar already done)"
tags: ["roadmap", "v1", "routers"]
sprint: ""
gh: null
---

# v1 integration /training: consolidate self_train.py + lora_eval.py + cloud_training.py into the training router (sidebar already done)

Parent: 20261101_140445_v1-route-map-land-the-7-pending-targets-models-souls-knowled (umbrella; cards cannot be updated via CLI so linkage lives here).
Route Map row /training (partial; verified 2026-11-01): nav-level consolidation ALREADY DONE — sidebar keeps /training + /training/runs; /auto-train, /self-train, /lora-eval pages already removed (headless). Remaining gap is routers: apps/api/server/routers/self_train.py + lora_eval.py + cloud_training.py still separate from apps/api/server/training/router.py; stale SIDEBAR_ICONS entries (/auto-train, /self-train, training sub-page icons) are shrinkable.
Work: consolidate the three routers into the training router/facade per Endpoint Rule, no behaviour change to headless flows, clean stale icon-map entries, run scripts/benchmark_slonet_training.py if the trainer path is touched.
Journey citation required at pickup: a ux/ flow in docs/UX_FLOWS.md (Page Rule test 1); no citation = bounce back before code.
Gates at pickup: vitest run of apps/web/lib/page-journey-gate.test.ts green; tsc exit 0; eslint 0 errors; page-baseline ratchet (any new page.tsx must be catalogued in apps/web/lib/page-baseline.json citing a ux/ journey; grandfathered/unmapped counts may only shrink; deleting a page = delete its entry + decrement GRANDFATHERED_CEILING); change-scoped tests via scripts/run_affected.sh; tests/test_layer_conformance.py green.
Surface ladder: cheapest surface that works — global banner < inline widget < chat mode (/chat?mode=) < plugin panel < operator page < journey page; a page is the most expensive rung. This card is a router/merge task: do NOT add a page.tsx unless the journey citation survives the ladder.

SEQUENCING: this card touches apps/web/lib/navigation.ts + apps/web/lib/page-baseline.json, shared with every other v1 route-map card — must run SEQUENTIAL, never in parallel with sibling route-map cards (Gen batches must sequence them).
Router merges are descriptor projections (Endpoint Rule): no hand-built feature routers; merged routes delegate to domain.<pkg> public facades only — 0 _internal imports (tests/test_layer_conformance.py enforces).

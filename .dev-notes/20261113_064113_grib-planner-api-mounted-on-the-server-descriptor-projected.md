---
id: 20261113_064113_grib-planner-api-mounted-on-the-server-descriptor-projected
status: todo
board: 20261113_064113_grib-planner-api-mounted-on-the-server-descriptor-projected
title: "grib planner API mounted on the server: descriptor-projected routes over PlannerStore, /planner page goes same-origin, retire the :8787 relay"
tags: ["area:api"]
sprint: ""
gh: null
---

# grib planner API mounted on the server: descriptor-projected routes over PlannerStore, /planner page goes same-origin, retire the :8787 relay

From session 2026-11-13 (user: 'how can we have our server process launch it on the same port as the web UI'). Today: apps/web/lib/planner-proxy.ts PLANNER_API_BASE defaults to http://127.0.0.1:8787 (the standalone `grib gui` stdlib sidecar); the only call site is the hand-built relay app/api/planner/transition/route.ts (hardened with x-slo-relay pin + origin-guard in card 20261108_142919 — the relay exists ONLY because :8787 is a foreign origin). RULING APPLIED at review: MOUNT, DO NOT SPAWN — one socket/one accept loop, so same-port = one process serving both route trees; spawning the sidecar + reverse proxy keeps two lifecycles, port stepping, and an auth-less stdlib server behind the public one. PLAN: (1) new routers/planner.py — ToolSpec descriptors (notes list/create/update/delete, board get/move, sync) via create_router → manifest + gen_ts_contracts regenerate; (2) handlers bind PlannerStore (the JOURNAL) — NOT the GUI's default md NoteStore, else the card-20261112_063528 revert bug returns through the GUI path (that fix shipped journal-side only: 1060548c6); (3) apps/web planner-client.ts/planner-proxy.ts switch to same-origin apiGet/apiPost (http-client rule), no raw fetch; (4) retire app/api/planner/transition after parity (grandfathered hand-built → deleted, not migrated); (5) NO new page — compose onto existing apps/web/app/(app)/planner/page.tsx (page rule); origin-guard/auth scopes unchanged (routes flow through normal middleware). MUST-NOT: touch packages/grib standalone `grib gui` (grandfathered, keeps :8787 default), break the origin-guard 403s, or add pages. GATES: check_contract --static baseline accounting, gen_router_manifest --check, gen_ts_contracts --check x3, /planner journey test, full grib suite 376, ruff+tsc+eslint clean. TDD RED first.

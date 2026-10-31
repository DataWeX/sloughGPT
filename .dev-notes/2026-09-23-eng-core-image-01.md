---
id: eng-core-image-01
status: done
priority: high
sprint: 2026-09
title: Shared Core (GSI) — one system image for app + web
body: |
  Treat system-state logic + endpoints as one shared kernel that both React
  Native (apps/api/server/routers/mobile.py BFF) and web (Next.js) ride;
  translate at the edge, never into the core.

  Delivered (all on merge-to-main):
  - docs/API.md "Shared Core Contract" (rules, core-stable endpoints table,
    state read-model, SSE frame schema, error taxonomy) + mobile plan note.
  - docs/plans/mobile-app-plan.md §0.1 API contract note.
  - system.py now reads executor state via public read-model:
    TrainingExecutor.max_workers / .job_count (no _max_workers/_jobs read).
  - routers/inference.py public facades build_session_metadata_index() and
    handle_chat(); mobile BFF offline sync + session list ride them instead
    of importing routers.inference._instance.
  - tests/contract/test_router_boundary.py: enforces routers never import
    another router's private _instance; pins the inference facades.
  - wired /phoneme router (was built, never registered) -> 51 routers.
  - api_keys.py kept (ApiKeyManager used by security/auth/workspaces).
  - benchmark_slonet all 7 sections PASS (fixed stale domain.slolib.gpu
    import, pre-existing on main).

  Verification: 71 passed (contract + system + server_api + mobile router),
  ruff clean, benchmark green at tip.
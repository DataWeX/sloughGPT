---
id: 20261001_bootstrap_batching_review
title: "Phase 3 review: /bootstrap boot fan-out batching (needs user review)"
status: todo
date: 2026-10-01
tags: [backend, performance, cors, design]
board_id: 342d729f-740e-413d-aad9-253cd1953439
board: 342d729f-740e-413d-aad9-253cd1953439
hash: 8ecd4b700b18d6ed2aa43ae8e5797a05e3c231881831a82d94bca7908758a0c6
---

# Phase 3 review: /bootstrap boot fan-out batching (needs user review)

Page load fires 11 parallel API calls (+ CORS OPTIONS preflights when
cross-origin) against a busy single-process backend; proposed fix was a
/bootstrap batch endpoint returning the boot fan-out in one response, plus
explicit CORS max_age on both layers. User requested review before building.
Acceptance: design reviewed by user, then card re-scoped.

SHARED WORKSPACE
- index updated: no index change required (design review only; /bootstrap would
  be a route on the existing API, not a new app/package)
- shared copy reused: root .venv + node_modules, no installs
- coordination named: none — re-scope after user review

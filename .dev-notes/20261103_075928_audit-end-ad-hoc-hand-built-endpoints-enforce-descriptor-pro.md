---
id: 20261103_075928_audit-end-ad-hoc-hand-built-endpoints-enforce-descriptor-pro
status: done
board: 20261103_075928_audit-end-ad-hoc-hand-built-endpoints-enforce-descriptor-pro
title: "Audit + end ad-hoc hand-built endpoints — enforce descriptor projection (user-ruled B, 2026-11-03)"
tags: ["contract", "backend", "audit"]
sprint: ""
gh: null
---

# Audit + end ad-hoc hand-built endpoints — enforce descriptor projection (user-ruled B, 2026-11-03)

User ruling 2026-11-03 (decode B): 'we keep hand-building and naming endpoints ad hoc — that's redundant, and we need to end that practice.' The Endpoint & Transport Rule (AGENTS.md) says endpoints are projections of ToolSpec-shaped descriptors (create_router(spec,...), generated manifest, generated TS contracts), but real code still hand-builds feature routers/HTTP routes outside the descriptor path. SCOPE: (1) census — grep every apiRouter/apiPost route + every FastAPI router not registered via create_router; classify descriptor-projected vs hand-built; (2) report the list with counts (which modules, how many routes); (3) migrate or card each offender (grandfather clause: migrate when next touched — but the user ruled END the practice, so decide per-router whether a migration sweep is warranted vs the grandfather clause); (4) add/extend a gate so a NEW hand-built route fails CI (descriptor parity gate already exists — check why it doesn't catch these). NOT in scope: god-endpoint retrofit, tool-tier crossings. Evidence to attach: census output + gate run. Related: card 20261102_110421 (ProcessEvent schema unification) proved the projection pattern works for non-ToolSpec contracts too.

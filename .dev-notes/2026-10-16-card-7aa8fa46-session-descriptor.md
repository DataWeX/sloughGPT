---
id: 7aa8fa46
title: "Session contract descriptor: declare session ops once (ToolSpec) -> agent tools + integration tests + UI"
status: doing
board_id: Main
date: 2026-10-16
tags: [contract, toolspec, session, testing-contract]
---

# Card 7aa8fa46 — session contract descriptor (claim)

Claimed 2026-10-16 by the foreman session for a Gene batch alongside
a7bb38fd (uxjourneys rehome land).

## Foreman pre-screen (done at claim time)

- Port-don't-rebuild: `git log --all --grep='session descriptor'` → only the
  card-creation board commit a29edfa56; `git log --all -S 'session_descriptor'`
  → nothing. No implementation exists on any branch — build fresh, projecting
  from existing infrastructure:
  - `ToolSpec` — `domain/agents/_internal/tools.py` (+ `domain/agents/__init__.py`)
  - HTTP projection emitter — `apps/api/server/infrastructure/contract.py`
    (ToolSpec/create_router/descriptor hits present)
  - TS emitter — `scripts/gen_ts_contracts.py` →
    `apps/web/lib/protocol/contracts.gen.ts` (freshness test
    tests/api-server/test_ts_contracts.py per AGENTS)
  - Spec doc — `docs/TRANSPORT_PROJECTIONS.md`, AGENTS 'Endpoint & Transport
    Rule' (core-first, descriptor-projected; no endpoint-per-feature)
- Territory: clean in domain/agents, apps/api/server/infrastructure,
  contracts.gen. Foreign dirt to avoid: packages/app-planner store.py +
  test_store.py (union-on-write, card 8c4b7927 says off-limits),
  apps/web/lib/chat-utils*.ts + apps/web/components/** (frontend session),
  routers/models.py, routers/tokens.py, routers/training/*,
  tests/api-server/test_tokens_router.py, tests/server/test_audit_instrumentation.py,
  tests/server/test_models_router.py.
- Surface-size data for the card's step-1 boundary call:
  routers/session.py = 261 lines, routers/chat.py = 180 lines; planner-side
  ops would route through the foreign-dirty store.py — prefer the chat/
  session surface unless planner forces it.

## Status

In progress — brief handed to Gene.

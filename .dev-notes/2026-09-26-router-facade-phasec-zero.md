---
title: Fix router domain-facade reach-ins — phase C (zero-out)
status: done
date: 2026-09-26
---

# Router facade reach-ins — phase C (Build Order #4)

## Task
Eliminate the remaining router `domain.*._internal` reach-ins: add missing
facade exports, swap import statements to public paths, tighten guard.

## Result (2026-09-26)
Commit `20a14db2c`, merged to main.
- 90 import statements across 20 routers now go through public facades.
- New facade exports (39 names): agents (MultiAgentOrchestrator,
  get_agent_run_store), cognition (ConsciousnessTrainer, TrainingConfig —
  lazy, torch-heavy), infrastructure (get_event_buffer, get_server_buffer,
  artifact_registry submodule), feedback (get_lora_evaluator — lazy),
  inference (get_engine, configure_api_provider — lazy), shell
  (InsFault/MemFault/Role eager; ShellREPL/get_bridge/render/sim — lazy),
  auth (Tenant/User/WorkspaceRepository, ROLE_PERMISSIONS).
- artifact_registry is NOT a root-level submodule → explicit eager export
  (stdlib-only, light).
- Guard reach-in baseline: 131 (origin) -> 105 (phase B) -> **2** (only
  known-dead workspaces dataset/knowledge sites, pending product decision).
- 5 test mock targets repointed from `_internal` to facade paths
  (agents.get_agent_system, infra.get_event_buffer,
  feedback.get_per_user_lora x3) — routers bind eager facade refs, so
  patching `_internal` no longer intercepts. Lazy-facade names
  (lora_evaluator, shell render/sim) still work via live __getattr__.
- Gates: facade probe 26/26, guard 2/2, ruff clean, full tests/server
  **1167 passed / 1 failed** (test_knowledge_router::test_search_files =
  pre-existing, card 063).

## Done
Build Order #4 phases A+B+C complete: routers no longer reach into
domain `_internal` (except 2 documented dead sites).

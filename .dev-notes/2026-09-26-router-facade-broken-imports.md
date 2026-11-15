---
id: 20260926_062_router_facade_broken_imports
status: done
title: "Build Order #4 (phase A): fix 7 broken domain-facade imports in routers + guard test"
tags: [backend, architecture, routers, facade]
---
# Card 062 — Build Order #4 phase A: broken facade imports

## Scope (phased)
Build Order #4 = "Wire routers — point routers at engines, not internals"
(131 `_internal` reach-ins across 22 routers + 7 fully-broken facade
imports, inventoried via AST scan 2026-09-26).

- **Phase A (this card)**: the 7 imports that raise ImportError at runtime
  (latent dead code paths hidden behind try/except) + a guard test so they
  can never regress.
- Phase B: training-domain `_internal` reach-ins (~35 sites: settings,
  dashboard, models, system, model_stack, self_train) → facade/engine.
- Phase C: remaining domains (settings._internal ×17,
  cognition.consciousness.personality ×16, shell, auth, memory, agents...).

## The 7 broken imports
1. `domain.training.TrainingOutcomeTracker` (health.py) — exists at
   `_internal.outcome_tracker` → add lazy facade export.
2. `domain.training.AdaptiveConfigEngine` (health.py) —
   `_internal.adaptive_config` → lazy export.
3. `domain.training.import_from_sou` (souls.py:181) —
   `_internal/slonet.py:4996` (sibling of already-exported export_to_sou).
4. `domain.infrastructure.emit_error_event` (infer.py:402) —
   `_internal/errors.py:521` (classify_exception already exported).
5. `domain.learner.enrich_with_knowledge` (inference.py:521) —
   `_internal/knowledge_augmenter.py:151` → eager facade entry.
6. `domain.models.update_personality_traits` (souls.py:391) —
   `_internal/provider/setup.py:312` (provider pkg already re-exports).
7. `domain.training.repository.TrainingRepository` (workspaces.py ×3) —
   **class does not exist anywhere**; the real store is
   `apps/api/server/training/job_store.py::JobStore` (`get_job_store()`,
   `list_by_workspace` -> list[dict], `delete`). Retarget + dict access.

## Guard
`apps/api/server/tests/test_router_domain_facade.py` — AST-scans every
router for `from domain.X import Y` and asserts each resolves via
importlib; plus asserts no `domain.*._internal` module path appears in a
**top-level** (module-scope) import (function-body reach-ins are phase B/C).

## Gates
- facade probe: all 7 names importable; workspaces cleanup uses JobStore
- ruff clean; pytest tests/server (subset + guard)

## Result (2026-09-26)
Commit `88380d8c7` on `feat/router-facade`, merged to main.
- All 7 broken facade imports fixed (probe 7/7 green).
- Guard test found **2 MORE dead imports** during development (generalized
  check to real `from X import Y` semantics):
  - `registry.py` `domain.infrastructure._internal.artifact_registry` —
    false alarm under exec-semantics (submodule import fallback) — passes.
  - `workspaces.py:1281` `domain.dataset._internal.repository` — **package
    `domain.dataset` never existed**; `workspaces.py:1299`
    `domain.learner._internal.knowledge.KnowledgeRepository` — never
    existed either (shim module has no such class; only
    `infrastructure/_internal/entity_repositories.KnowledgeRepository`
    exists and has NO workspace scoping anywhere in the codebase).
    These two workspace search/cleanup blocks are inert behind
    try/except and are **allowlisted as known-dead pending a product
    decision** (add workspace scoping or drop the blocks).
- Gates: facade probe 7/7, guard 2/2, ruff clean, affected router suites
  132/132, full `tests/server` **1166 passed / 2 failed**.
- The 2 failures (`test_knowledge_router::test_search_files` 400
  "Path not found: routers", `::test_stats_with_many_facts_terminates`
  hang) **reproduce on clean baseline with changes stashed** →
  pre-existing, NOT this card. Follow-up candidate (new card).

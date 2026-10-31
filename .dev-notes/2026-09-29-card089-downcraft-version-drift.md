---
id: card089
title: "Card 089 — downcraft version drift: env 0.2.0 (flat) vs repo 0.3.0 (nested)"
status: todo
priority: medium
created: 2026-09-29
source: card 083 verification (red test_router_domain_facade guard)
---

## Problem

- conda env has `downcraft 0.2.0` (flat: `downloader.py`, `state.py`,
  `verify.py`) in site-packages since 2026-09-18.
- repo has `packages/downcraft` **v0.3.0** nested (`download/{state,http,verify}.py`),
  which all code imports target (`domain/infrastructure/_internal/hf_hub.py:38-40`,
  `apps/api/server/routers/models.py:955`).
- App survives (`main.py:49` inserts `packages/downcraft` at position 0);
  root conftest now does the same (card 083 fix). Contexts WITHOUT either
  bootstrap (standalone scripts like `scripts/setup_android_sdk.py:52`,
  ad-hoc `python -c`) resolve site-packages 0.2.0 → `downcraft.download`
  ModuleNotFoundError.

## Decision needed (user)

- Option A: `pip install -e packages/downcraft` into the shared conda env
  (replaces 0.2.0 with 0.3.0 editable — one shared copy, matches policy).
- Option B: `pip uninstall downcraft` — rely on repo path bootstraps only.
- Option C: leave as-is — conftest + main.py cover the two real runtimes.

## Done when

Install policy chosen and executed; `python -c "from downcraft.download import
state"` works with plain env python (no PYTHONPATH); guard test stays green.

---
id: 20260924_041_retarget_plural-imports
title: Dual-tree phase 1 — retarget domains.* → domain.* where singular is canonical
status: done
sprint: current
gh: ""
body: >
  Retarget plural imports to singular for modules that already have real code
  under domain/. Do not touch imports whose only real implementation still lives
  in packages/core-py/domains (pugqeep, shell addons/cmds, quant_core, slnc, cache,
  deployment, training_bridge) — those need file migration first.
---

# Dual-tree phase 1 — retarget plural imports

## Scope
- [x] Inventory domains.* imports; classify retarget-safe vs must-migrate-first
- [x] Retarget safe set (tests + prod)
- [x] Gates green (targeted tests + scatter + ruff)
- [x] Merge to main

## Result
- Commit `5318fff2c` merged FF to `main` (tip after merge = 5318fff2c).
- **20 files, 44 insertions / 43 deletions** — quant_core left on plural (real ctypes wrapper is plural-only); pugqeep/shell/ml_infrastructure must-migrate left alone.
- Prod facades: `domain/infrastructure/{cancel_manager,server_state}` → `_internal` twins.
- Prod lazy imports: slo_embedder + multimodal engine → `domain.training._internal.slonet`; avion → `domain.inference._internal.slonet_provider`.
- 15 test modules retargeted to canonical `_internal` so monkeypatch targets match production module objects.
- **Gates**: ruff clean; scatter 5/5; **43 vs 62 baseline failures — 0 new, 19 fixed** (multimodal engine/manager + truth_maintainer clips). Import probe 22/22 (avion/arken pre-existing).
- Remaining plural imports (out of scope): pugqeep, shell cmds, quant_core, slnc, cache, deployment, ml_infrastructure.benchmarking, examples, colab string.

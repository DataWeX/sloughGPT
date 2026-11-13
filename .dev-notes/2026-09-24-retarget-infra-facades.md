---
id: 20260924_042_retarget_infra_facades
title: Dual-tree phase 2 — retarget 40 infrastructure facades to _internal twins
status: done
sprint: current
gh: ""
body: >
  Flip domain/infrastructure/*.py facades from domains.X to
  domain.infrastructure._internal.X where a real non-alias twin exists (~40 files).
  Leave pugqeep (twin aliases plural). Does not migrate plural-only real files.
---

# Dual-tree phase 2 — infrastructure facades

## Scope
- [x] Flip ~40 safe facades to _internal
- [x] Gates green (targeted + scatter + ruff)
- [x] Merge to main

## Result
- Flipped **42** facades `from domains.X import *` → `from domain.infrastructure._internal.X import *` (skipped `pugqeep` — twin still aliases plural).
- On this branch also flipped `cancel_manager` + `server_state` (phase 1 not ancestor of `feat/training-resume-ux`).
- Worktree facades **byte-identical** to `feat/retarget-infra-facades` @ `a0fbaf72a`.
- **Gates**: ruff clean; import probe **43/43**; scatter **5/5** (from main); core smoke **335** + earlier batches **478/265/153/34/125** green; apps/api registry+dashboard+process_guard **34** green.
- **Pre-existing failures (0 new, same with flip stashed)**: `test_session_core` (10 — wrong patch path vs `_internal`), `test_memory_pressure` classify (1), `test_training_queue` `domain.training.chat_trainer` missing (15 — module is `_internal.chat_trainer`), `test_weight_loader` (6).
- Remaining plural-only: `pugqeep` facade (by design).

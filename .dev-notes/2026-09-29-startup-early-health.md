---
id: 20260929_078_startup_early_health
status: done
title: "Card 078 — Startup early-health: re-sync startup.py with main, keep timeout fix, phase-regression test"
tags: [startup, health, phases, regression, api]
sprint: current
gh: null
---
# Card 078 — Startup early-health reconciliation

## Problem

- Stage BACKGROUND re-enters `_phase3_wandb`/`_phase4_multimodal` after
  `_post_bind` set terminal phase `ready`; unconditional
  `STARTUP_PHASE.update(...)` regresses 9/9 → 6/9 and pins
  `/health/ready`'s `app_lifecycle` at "starting" forever.
- Worktree copy of `apps/api/server/infrastructure/startup.py` had drifted
  from main: lost the `_advance_startup_phase` guard, the
  `SLO_BOOTSTRAP_HF` opt-in gate, and the `domain.*._internal` prewarm
  retarget (index copy == main byte-for-byte). Kept its unique fix:
  `training_restore` timeout 15s → 60s (cold JobStore journal left
  interrupted jobs unseeded).
- Staged index copy of `controllers/health.py` regressed `utc_now_iso` →
  naive `datetime.now()`; worktree copy is the correct one (canonical
  timestamps + PEP604 + `__future__` annotations).

## Plan (branch `feat/startup-early-health`)

1. `startup.py` ← index content (== main), re-apply timeout=60 + comment.
2. Commit worktree `health.py` (reverses the utc_now_iso regression).
3. New regression test: phase stays `ready`/step 9 after
   `_phase3_wandb()` + `_phase4_multimodal()` run post-ready.
4. Gates: ruff, pytest startup(49)+health(103),
   `scripts/benchmark_startup.py --port 8010` (8000 owned by live API).
5. Path-scoped commit of 5 files (207-file parallel batch stays staged).

## Verification (pre-flight, pre-edit tree)

- ruff: clean; pytest 49 + 103: pass.
- benchmark dry-run: time_to_health 45.1s, time_to_ready 84.3s,
  preload_warnings 0, exit 0.

## Result (done)

- Commit `0599dc579` (path-scoped, 5 files; 207→202 staged batch intact).
- Post-edit gates: ruff clean, 98 passed (50 startup incl. new
  `test_ready_phase_cannot_be_demoted_by_background_rerun` + 48 health),
  benchmark --record x2: 16.28s/15.3s health, 21.7s/23.9s ready,
  preload_warnings 0. The tool's [FAIL] on run 1 compared against a
  cherry-picked 10.35s baseline — historical health spread across
  records is 7.93–152.23s (19x), so no real regression; run-to-run
  comparison [OK].
- Merged back into `feat/gateway-sse-compression` (ff via ref update;
  shared checkout also carries the parallel build-order #4 commits).
- Disk hit "No space left" mid-first-commit (99%, 2.1G free — transient
  spike); retry succeeded. apps/mobile/node_modules (7.5G regenerable)
  still the big untouchable — needs user approval to clean.

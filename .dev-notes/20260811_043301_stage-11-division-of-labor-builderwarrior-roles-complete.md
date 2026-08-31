---
id: 20260811_043301_stage-11-division-of-labor-builderwarrior-roles-complete
title: Stage 11 division of labor — Builder/Warrior roles complete
status: done
tags: world-realm,stage11,specialization
created: 2026-08-11T04:33:01.968689+00:00
---

Stage 11 division of labor — Builder/Warrior roles complete

Stage 11 (division of labor) verified complete and documented.

IMPLEMENTATION (already present in evolution.py, 1649 lines): heritable role posture via perceptron_role (body-input -> 1 gate), gate < role_gate_threshold (0.5) = Builder, >= = Warrior. Builder banks role_deposit_fraction (0.1) of surplus above start_energy into own nest within nest_use_radius when standing on it; Warrior raids foreign nest within territory_radius at min(role_raid_fraction (0.5) x nest_draw_rate, bank) even when not hungry. Both pure transfers (conservation invariant holds), both land in same tick's honest net reward. Role brain zero-built when channel off; weights draw from dedicated _ROLE_RNG_SEED=0x1E09 stream so the four behavior brains stay bit-identical (locked proofs intact).

KEY BUG FOUND & FIXED: benchmark_specialization (evolution.py line 1131) did NOT set write_energy_scale, so writes never carried nest_seed_energy=3.0, no nests seeded, role acts could never fire. Fixed via replace(base, ..., write_energy_scale=10.0) mirroring benchmark_territoriality/lifecycle. Verified: deposits fire, raids fire every gen, 8 nests.

CLI --specialize verified live (--social-pools 2, seed 7): raids every gen, specialization beats control by gen 8 (40.92 vs 9.85). Note: CLI flag is --social-pools (no --organic-pools).

TESTS: test_specialization 28 pass, test_lifecycle 58 (combined), test_evolution 109 (incl. locked proofs + six role keys in test_run_structure), test_simulation + test_world_driver 141, shell REPL/runtime/state/VFS/pane suites pass.

DOCS: Stage 11 section added to docs/WORLD_REALM.md after Stage 10.

Pre-existing env issues (unrelated): fastapi not installed -> 32 router tests + test_auto_train_helpers fail at collection; full suite > 30 min (heavy VM tests). Run targeted suites only.
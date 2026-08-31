---
id: 20260809_144352_world-realm-headless-driver
title: world-realm-headless-driver
status: done
tags: world-realm,shell,cli
created: 2026-08-09T14:43:52.813625+00:00
---

world-realm-headless-driver

Built WorldDriver observability harness + CLI wrapper for the world realm.

- domains/shell/world_driver.py: WorldDriver (build world, run ticks via real Simulation loop, snapshot energy/material/signal, energy_ledger, conservation_report — total grid+entity energy must never increase; run_evolution delegates to EvolutionEngine). Material names derived programmatically from MATERIAL_* constants (no hardcoded display table). CLI: python3 -m domains.shell.world_driver --grid W,H,D --seed N --ticks N --every N [--evolution --generations N --population N --ticks-per-gen N].
- tests/test_world_driver.py: 21 tests (naming, grid parsing, empty-vs-terrain pops, snapshot shape, tick cadence, ledger consistency, conservation monotonic + violation detection, seed determinism, evolution summary, CLI output + bad-arg path).
- docs/WORLD_REALM.md: World Driver + 21 Tests rows added to Built and Working; Standalone section gains the headless CLI example.
- Verified: py_compile clean; full targeted scope 533/533 pass (512 prior + 21 new), exit 0; CLI smoke run on 16x8x16 reports clean conservation (total 1432.7 -> 1291.4 over 10 ticks, monotonic), terrain materials populated, babies alive. pycache cleared. Board sync pending.
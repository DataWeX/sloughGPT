---
id: 20260809_093725_episodic-memory-energy-conservation-fix
title: Episodic Memory + Energy Conservation Fix
status: done
tags: shell,world-realm,memory
created: 2026-08-09T09:37:25.327656+00:00
---

Episodic Memory + Energy Conservation Fix

Completed: Episodic memory ring buffer + energy conservation fix.
- domains/shell/memory.py: Episode dataclass + EpisodicMemory (record/recall by k or by reward, mean_reward, stats, chronological order across ring wrap, capacity<1 raises). 17 tests in tests/test_memory.py.
- simulation.py wiring: WorldParams memory_capacity=64/memory_lookback=5; SimBaby.memory + recall_memories(); learn() records episode (features/action/reward/tick); info() exposes memory stats; SimScene.babies property.
- ROOT-CAUSE FIX (flaky tests, ~22%): cell writes created energy from nothing (deposit was free), so random babies could farm organic and outpace drain; test_step_reduces_energy_over_time and test_step_returns_results failed intermittently and np.random.seed(1) leaked state across tests. Simulation.step now funds deposits from the baby's own energy (conservation: deposits are transfers, not creation). write_energy_scale WorldParams param (default 1.0, was hardcoded 5.0) keeps the world survivable so the GA population no longer collapses (alive 10->9 over 6 gens, best fitness ~189 honest vs 263 exploit-inflated before).
- Verified: 429 passed (memory 17 + simulation 117 + evolution 20 + shell repl 275), simulation suite stable 5/5 stress runs, GA deterministic per seed.
- docs/WORLD_REALM.md: Memory out of Not Built, EpisodicMemory + Evolution rows in What Exists Now, funded-write energy economy documented, memory sections updated.
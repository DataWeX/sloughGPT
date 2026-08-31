---
id: 20260809_235111_stage-7-durable-structures-nests
title: Stage 7 Durable Structures (Nests)
status: done
tags: core,world-engine,stage7
created: 2026-08-09T23:51:11.870342+00:00
---

Stage 7 Durable Structures (Nests)

Completed Stage 7 first wave: durable nest structures.

Design (all opt-in via structure_enabled=False to keep locked selection proofs and their RNG streams exact): nest_radius=2.0, nest_seed_energy=3.0, nest_draw_rate=1.0, nest_use_radius=2.0, nest_decay=0.002, max_nests=8.

Implementation in domains/shell/simulation.py: Nest dataclass (id, position, stored_energy, owner_group_id, alive, to_dict/from_dict); SimScene.nests list + _next_nest_id; methods nearest_nest(point, radius, group_id=None), route_build(action, baby), update_nests(), draw_nest(baby). Simulation.step order: deliver messages -> update_nests -> world compute -> per baby (perceive incl. nests -> apply_action -> route_build -> move/message/social/learn -> absorb -> draw_nest -> drain). Result dicts carry nested/seeded/drawn; summary() has total_nested, total_drawn, nests_built; SimScene.info() has nests, nest_energy.

Rules: deposit within nest_radius feeds nearest nest bank (transfer, cell zeroed); deposit >= nest_seed_energy far from any nest seeds a new nest owned by writer's tribe; starving baby (below start energy) draws up to nest_draw_rate from own tribe's nearest nest within nest_use_radius; nests decay per tick and erode when empty; max_nests cap bounds territory. Feed/seed/draw are all transfers -> conservation invariant holds.

route_build uses assign-then-increment for nest ids (first nest id=1), mirroring SimBaby.

world_driver.py: snapshot keys nests + nest_energy; energy_ledger total = grid + entities + nests; CLI --structures; tick-table nest_energy column. tests/test_world_driver.py snapshot/ledger contracts updated.

Tests: 14 new tests in TestNests (test_evolution.py), all passing. Full regression: 104 tests across test_evolution + test_world_driver pass. Locked benchmarks re-verified: benchmark_social(seed=1) group_weight=0.5 ind_coop=0.0 grp_coop=0.03125 ind_contest=0.18229166.. grp_contest=0.02604166..; benchmark_emergence seeds 1-5 -> True True False False True (result key e['evolved']['best_fitness']).

Docs: docs/WORLD_REALM.md updated (Stage 7 ladder BUILT, Built-and-Working rows for nests, test counts 68->82, World Driver --structures, WorldParams block, emergent-behaviors item 6, new nests detail section; Not Built now only Stage 7 Culture/teaching).
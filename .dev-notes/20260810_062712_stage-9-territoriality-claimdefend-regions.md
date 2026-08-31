---
id: 20260810_062712_stage-9-territoriality-claimdefend-regions
title: Stage 9 territoriality: claim/defend regions
status: done
tags: world-realm,stage9,territoriality
created: 2026-08-10T06:27:12.112637+00:00
---

Stage 9 territoriality: claim/defend regions

Stage 9 territoriality close-out complete.

- simulation.py: territory channel merged; deduplicated doubled result-dict defense fields + verbose terr/defend token; fixed _push_away sign bug (shove now moves the trespasser away from the defender, per its docstring).
- evolution.py: dedicated _TERRITORY_RNG_SEED=0x1E07 stream drawn last (four behavior brains stay bit-identical with channel off); Genome territory tensor + apply_to + mutate routing; social defend_rate; benchmark_territoriality (defense-on vs defense-off arms on same grouped world, territoriality_emerged verdict).
- world_driver.py: --territory flag + benchmark print block.
- tests: test_territoriality.py (26 tests) green; predation (23) + world_driver (22) unchanged green; 71 total.
- docs/WORLD_REALM.md: Stage 9 section + emergent #1 marked Built + driver/tests rows already present; annotated emergent Territoriality entry.
- Verified: --territory benchmark run (defenses fire, territoriality_emerged=yes at 8 gens); py_compile clean.
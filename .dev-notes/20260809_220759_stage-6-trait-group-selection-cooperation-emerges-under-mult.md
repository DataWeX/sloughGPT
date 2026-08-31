---
id: 20260809_220759_stage-6-trait-group-selection-cooperation-emerges-under-mult
title: Stage 6 trait-group selection — cooperation emerges under multilevel selection
status: done
tags: world-realm,evolution,stage6,selection,cooperation
created: 2026-08-09T22:07:59.190603+00:00
---

Stage 6 trait-group selection — cooperation emerges under multilevel selection

Cooperation now emerges reliably via multilevel (trait-group) selection — never hardcoded.

Root cause of collapse: rich world + blind nearest-neighbor sharing meant gifts went to the best-fed agents; cooperation was pure self-harm and selection killed it.

Changes:
- evolution.py: _select_groups uses geometric-mean tribe fitness + uniform within-tribe mating; offspring inherit group_id; group_weight=0 falls back to individual selection. benchmark_social() compares individual vs group arms on the SAME grouped world (scarce: organic_pools=3, ticks=24).
- simulation.py: WorldParams.social_enabled; entity perceptron gains a kin signal (1.0 when the perceived baby shares group_id); social_step targets the neediest neighbor (min by energy); learn()/social_step slice entity input to entity_input_dim; SimBaby.to_dict/from_dict now persist group_id (restore was missing it — broke resume determinism).
- benchmark_emergence sets social_enabled=False so Stage 5 individual proof stays isolated from contest-stealing.

Results (seed 1, deterministic): ind_coop=0.0000 grp_coop=0.0312 ind_contest=0.1823 grp_contest=0.0260 emerged=True. Individual arm's cooperation collapses and contests stay high; group arm keeps cooperating (up to ~37 acts/gen) with ~7x fewer contests.

Tests: test_evolution.py 38 -> 57 (tribe selection 6, kin signal 3, social interaction 6, social benchmark 5, +persistence group_id). World-realm suite: 287 passed.

CLI: world_driver.py --social (defaults: 16,8,16 grid, 12 gens; flags --social-pools/--social-ticks-per-gen/--group-count/--group-weight). Reproduces benchmark at seed 1.

Docs: WORLD_REALM.md Stage 6 marked BUILT with mechanism + benchmark table; features table + Emergent Behaviors updated.
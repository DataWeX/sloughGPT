---
id: 20260810_130442_stage-9-territoriality-test-suite-green-docs
title: Stage 9 territoriality: test suite green + docs
status: done
tags: world-realm,stage-9,tests
created: 2026-08-10T13:04:42.214571+00:00
---

Stage 9 territoriality: test suite green + docs

Delivered test_territoriality.py (26 tests) covering the territory brain, defend mechanics, scene eviction semantics, persistence, learning, and evolution/benchmark invariants. The implementation was refactored mid-stream: the toll became defend_take_fraction (a share of the trespasser's energy, capped non-lethal), and the shove became _push_away (one defend_push cell from the defender) — the suite tracks both, including the geometric repeated-toll drain and that learning drifts the move gate after tick 1 (scene energy accounting uses max_ticks=1). All 195 tests green: 26 territoriality + 23 predation + 22 world_driver + 124 simulation. docs/WORLD_REALM.md now documents Stage 9 (claim/defend, benchmark, --territory CLI).
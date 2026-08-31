---
id: 20260810_020714_stage-7-culture-cross-agent-teaching-lateral-behavior-transm
title: Stage 7 Culture: cross-agent teaching (lateral behavior transmission)
status: done
tags: world-realm,stage7,teaching,culture
created: 2026-08-10T02:07:14.647539+00:00
---

Stage 7 Culture: cross-agent teaching (lateral behavior transmission)

Stage 7 Culture (lateral teaching) locked: 3/5 emerged, mean +9.24 (seeds 2,3,5 win; 1,4 lose) on benchmark_culture, cap=1 episode copy.

Mechanics: perceptron_teach (entity-input -> 1 gate, RNG-neutral) per baby; gate clears teach_gate_threshold (0.5) -> lesson at gate amplitude; student behavior weights blend toward teacher by teach_weight_blend*amp; up to teach_memotype_cap (default 1) best-reward episodes copied; target = neediest same-group neighbor in teach_range; teacher pays teach_cost*amp into its same-tick net reward. WorldParams: teaching_enabled=False (off by default), teach_cost=0.5, teach_range=5.0, teach_gate_threshold=0.5, teach_weight_blend=0.1, teach_memotype_cap=1. cap floor removed in teach() so cap=0 disables episode transfer (true off-switch).

Root cause of earlier collapse fixed: delta-rule sign bug in Perceptron.update (negative lr double-flip) and reward was the uniform -0.5 see_cost (absorption measured after delta capture). Reward is now the honest net tick delta at step 8b (full energy flow incl. movement, writes, social transfers, teaching cost, absorption, nest draw, passive drain). Sign rule: gains reinforce, losses weaken.

Per-seed: seed1 control/culture 131.25/120.85 (-10.4 F), seed2 96.46/139.04 (+42.57 T), seed3 154.98/177.44 (+22.46 T), seed4 181.62/88.6 (-93.02 F), seed5 79.86/164.47 (+84.61 T). Teaching amplifies lineage quality either way (seed4's culture lineage drifted to aggression and teaching spread it). Emergence + social locks unchanged (emergence 1-5 True True False False True).

Tests: 14 new in tests/test_evolution.py (delta-rule sign x3, teaching x7, culture benchmark x4); suite 82->96. test_run_structure expected keys updated (lessons, teach_rate). Docs: WORLD_REALM.md Stage 7 row, params listing, design table (teaching row, 96 Tests row), World Driver row (--culture), reward narrative (honest net delta + two fixed defects), Evolution section. CLI: --culture. Serialization carries teach weights only when channel on.
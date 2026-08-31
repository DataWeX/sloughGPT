---
id: 20260809_201317_stage-5-cognition-emergence-proof-evolution-driven-deep-brai
title: Stage 5 cognition emergence proof — evolution-driven deep brain
status: done
tags: world-realm,evolution,stage5,cognition
created: 2026-08-09T20:13:17.606388+00:00
---

Stage 5 cognition emergence proof — evolution-driven deep brain

Stage 5 (COGNITION FROM SCRATCH) built per docs/WORLD_REALM.md. Mechanism: evolution-driven emergence; in-life delta learning is optional/legacy.

Key decisions/root causes:
- Noise collapse: random food layout + random spawn every generation made fitness measure luck, not genes; selection amplified noise and the population collapsed (~12 vs ~100 baseline). Fixed with a fixed-environment mode: deterministic terrain (world_seed), food pools re-seeded identically each generation, and fixed spawn positions (shared spawn for all genomes so crossover mixing stays coherent).
- Breeding bug: EvolutionEngine._select bred children via crossover(winner, random genome) with k fully random — half of every child's alleles were random. Now both parents are tournament winners.
- Delta learning is destructive under selection: uniform scalar error reinforces the self-funding write that drains the baby; excluded from the benchmark (learning_enabled=False), documented as optional.

Delivered:
- Perceptron deep brain (optional hidden projection, no backprop), movement (perceptron_move + sign(gate-0.5) gates + move_cost + starvation stay-put), Genome covers move tensors, run_frozen() no-selection baseline, benchmark_emergence() deterministic A/B.
- Emergence proof: evolved last-gen mean beats frozen-random on 5/5 seeds at hidden=0 (mean 116.6 vs 87.0) and hidden=4 (120.5 vs 83.9); gen-1 arms identical.
- CLI: world_driver --emergence [--hidden-units N].
- Tests: 6 new in test_evolution.py + 1 in test_world_driver.py; full 7-file world suite green.
- Docs: docs/WORLD_REALM.md Stage 5 marked BUILT, mechanism + proof documented, Not Built table updated (Stage 6 + long-term memory).
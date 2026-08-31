---
id: 20260809_031947_multi-agent-social-simulation-babies-cooperate-and-compete
title: Multi-agent social simulation — babies cooperate and compete
status: done
tags: simulation,multi-agent,world-realm
created: 2026-08-09T03:19:47.942892+00:00
---

Multi-agent social simulation — babies cooperate and compete

Built the Multi-Agent capability listed as 'Not Built' in docs/WORLD_REALM.md. Babies already shared a world but could not perceive each other; Perception.nearby_entities and perceptron_entity existed but were never populated.

Changes (packages/core-py/domains/shell/simulation.py):
- WorldParams: social_radius, share_fraction, contest_take, contest_threshold, cooperate_threshold
- SimScene.nearby_babies(position, radius, exclude_id) — alive neighbors sorted by distance
- SimBaby.perceive(world, babies) now fills nearby_entities (id, type, energy, distance, angle)
- SimBaby.share_energy(other) — cooperative, transfers fraction of surplus (conserved)
- SimBaby.contest_energy(other) — competitive, takes capped amount from weaker agent (conserved)
- SimBaby.social_step(other) — perceptron-driven decision: cooperate gate + surplus, contest gate + weaker
- SimBaby.learn() now also trains perceptron_entity from nearest perceived neighbor
- Simulation.step() passes alive babies to perceive and runs a social step per baby; results carry social_act/social_energy
- Simulation.summary() adds cooperations, contests, social_energy_moved

Tests: +27 in test_simulation.py (TestSceneNearbyBabies 6, TestSocialPerception 4, TestSocialMechanics 6, TestSocialStep 6, TestSimulationSocial 4). 112 passed (was 85).

Verified end-to-end: two cooperation-gated babies ran 20 ticks → 38 cooperate acts, 474.4 energy redistributed, both survived.

docs/WORLD_REALM.md: Multi-Agent moved from 'Not Built' to 'Built and Working'; SimBaby/SimScene/Simulation rows updated; test count 58→112.

Full python suite (packages/core-py/tests + tests/server) running detached; test_simulation.py fully green.
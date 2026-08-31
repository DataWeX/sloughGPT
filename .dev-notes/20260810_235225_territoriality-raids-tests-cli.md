---
id: 20260810_235225_territoriality-raids-tests-cli
title: Territoriality raids: tests + CLI
status: done
tags: core-py,territoriality,raid
created: 2026-08-10T23:52:25.242494+00:00
---

Territoriality raids: tests + CLI

Added raid support to Stage 9 territoriality: a hungry baby standing on foreign ground (within territory_radius of a rival nest) drains the bank (nest_draw_rate/tick, transfer not creation), the value defense protects. Implemented raid_nest in simulation.py, raided field in tick log, raids/raid_energy_moved in summary(), history rows and benchmark_territoriality verdict in evolution.py, and raids columns in world_driver --territory. Added TestRaid class (11 tests) covering: channel-off no-op, hunger requirement, own-bank protection, foreign-ground requirement, nearest-nest targeting, transfer conservation, gap/bank capping, scene summary recording, off-channel scene, and exact energy flow (including nest_decay ordering). Fixed stale history key-set assertion in test_evolution.py::test_run_structure (added defenses/defend_rate/defend_energy_moved/raids/raid_energy_moved/nests_built). All tests pass; CLI verified.
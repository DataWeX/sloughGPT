---
id: 20260809_114212_scene-persistence-saveload-the-world
title: Scene persistence — save/load the world
status: done
tags: world-realm,persistence
created: 2026-08-09T11:42:12.457463+00:00
---

Scene persistence — save/load the world

Added JSON-safe scene snapshots: WorldGrid/Entity/Perceptron/SimBaby/SimScene to_dict+from_dict in simulation.py, EpisodicMemory to_dict+from_dict (exact ring-buffer incl. head) in memory.py. Restore is RNG-neutral (resume == uninterrupted run, verified bit-for-bit), id counter continues past restored ids so spawns never collide, JSON dump/load round-trips. 22 new tests in tests/test_scene_persistence.py. 468 world-realm tests pass (446 prior + 22 new); simulation+evolution stress 5/5 stable. WORLD_REALM.md updated (component table, emergent behavior #6, Persistence section).
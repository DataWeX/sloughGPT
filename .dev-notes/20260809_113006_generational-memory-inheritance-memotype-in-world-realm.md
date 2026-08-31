---
id: 20260809_113006_generational-memory-inheritance-memotype-in-world-realm
title: Generational memory inheritance (memotype) in world realm
status: done
tags: shell,world-realm,memory,evolution
created: 2026-08-09T11:30:06.481254+00:00
---

Generational memory inheritance (memotype) in world realm

Closed the documented gap 'Generations do not inherit experience - only weights' (WORLD_REALM.md). Genome now carries a memotype: from_baby() consolidates the parent's top-reward episodes (capped at new WorldParams.memory_inherit=8) into serializable dicts; crossover() inherits the winning lineage's memories wholesale; mutate() reshapes weights but never memories; apply_to() seeds offspring episodic memory at birth. EvolutionEngine.run() now reports inherited_episodes. Memory survives death through the lineage. 12 new tests (TestMemoryInheritance): consolidation order/capping/detachment, JSON-serializability, crossover/mutation behavior, apply_to seeding, selection propagation, generation-transition survival, run() observability. Verified: 446/446 scope tests (evolution 32 + memory 22 + simulation 117 + shell_repl 275); simulation+evolution stress 5/5 stable. Docs: WORLD_REALM.md limits table, memory section, Evolution section, component table, evolution test count 20->32.
---
id: 20260810_051845_stage-7-long-term-memory-world-reservoir-complete
title: Stage 7 long-term memory: world reservoir complete
status: done
tags: world-realm,stage7,memory
created: 2026-08-10T05:18:45.183446+00:00
---

Stage 7 long-term memory: world reservoir complete

Landed the final Stage 7 roadmap item (Infinite Long-Term Memory). The implementation already existed in code (WorldMemory in memory.py, memory_enabled/memory_deposit/memory_seed in WorldParams, SimScene.add_baby seeding + deposit_memory, evolution engine reservoir shared across generations with in-sim death deposits and survivor boundary deposits, scene snapshot serialization, run() memory_size/memory_seeds_total reporting, --memory CLI + benchmark_memory). What was missing was tests, docs, and verification.

Verification and closure:
- Added 18 WorldMemory unit + scene integration tests in tests/test_memory.py (append-only never evicts, record stamps group/donor, consolidate top-reward + zero-cap no-op, recall by reward/recent/group, stats, lossless to_dict/from_dict, reservoir off by default, enabled scene creates reservoir, deposit consolidation, newborn seeding, no-reservoir no-seed guard, scene snapshot preserves reservoir).
- Added 7 evolution reservoir tests in tests/test_evolution.py (off-by-default guard, enabled engine carries reservoir, reservoir grows monotonically across generations, newborns seeded, deposits stamped tribe+donor, determinism with memory on, deposit cap). test_evolution.py 96 -> 103; test_memory.py 22 -> 40.
- Fixed test_run_structure exact-keys contract to include the two new history keys (memory_size, memory_seeds).
- Verified --memory CLI end-to-end: reservoir 32 -> 64 across 2 gens, seeds given = 16, ~1s.
- Updated docs/WORLD_REALM.md: removed the last 'Not Built' row (Infinite Long-Term Memory), updated Stage 7 block to final wave, added World Memory Built-table row + mechanism section + params (memory_enabled/deposit/seed), updated World Driver row + test counts.
- Full shell suite green: test_evolution (103) + test_memory (40) + test_simulation + test_world_driver + test_scene_persistence + test_signal_communication + test_environment. Locked culture benchmark (5 tests incl. 3/5 verdict across seeds 1-5) intact.

Not committed (no commit instruction given). Working tree still carries prior-session dirty changes (dataset flow, frontend overhaul, ~50 new test files, deleted cypress file).
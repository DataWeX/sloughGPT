---
id: 20260813_002631_memory-consolidation-triggers-cli-auto-maintenance
title: Memory consolidation triggers: CLI + auto-maintenance
status: done
tags: memory,cli,infrastructure
created: 2026-08-13T00:26:31.888291+00:00
---

Memory consolidation triggers: CLI + auto-maintenance

Session 8. Wired the built-but-orphaned memory.consolidate task into the running system with two triggers.

CLI: added 'sloughgpt memory consolidate [--threshold F]' (commands/memory.py + cli.py). Runs the same plan_consolidation planner as the task handler: groups facts by topic, deletes shorter near-duplicates (n-gram cosine >= threshold, default MemoryConfig.consolidation_threshold 0.85), reports removed/kept. Verified live: exits 0, prints threshold + kept count against the real store.

Automatic maintenance: new domains/memory/maintenance.py - maintenance_tick() enqueues one memory.consolidate task via submit_memory_consolidate; run_memory_maintenance() is the asyncio loop (sleep interval then tick); start/stop_memory_maintenance() manage the background task idempotently. Wired into apps/api/server/infrastructure/startup.py: started in _phase_task_queue (after register_memory_handlers), stopped in _shutdown_task_queue (before queue stop). MemoryConfig grew maintenance_interval_minutes (env SLO_MEMORY_MAINTENANCE_INTERVAL_MINUTES, default 60; 0 disables). Docs: ENVIRONMENT.md + memory_config.py table row.

Tests: packages/core-py/tests/test_memory_maintenance.py (10: tick enqueues, noop when disabled/interval-zero, error swallowed, loop start/stop ticks periodically, config wiring). CLI TestConsolidate x3 (merge at 0.80, no-merge at default 0.85, empty store). Fixed 2 stale startup phase tests (test_startup_routers.py) - _phase_task_queue now also registers memory handlers + starts scheduler; patched the new calls, assert both run even when queue init fails.

Verification: 122 memory-surface tests pass, 49 startup tests pass, 93 CLI tests pass, py_compile clean, pycache cleared.
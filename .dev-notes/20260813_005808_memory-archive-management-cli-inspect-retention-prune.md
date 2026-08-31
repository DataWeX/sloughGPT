---
id: 20260813_005808_memory-archive-management-cli-inspect-retention-prune
title: Memory archive management: CLI inspect + retention prune
status: done
tags: memory,task-queue,cli
created: 2026-08-13T00:58:08.759582+00:00
---

Memory archive management: CLI inspect + retention prune

Session 9. Archive management for the task-backed provenance store (facts.jsonl).

CORE (task_memory.py): _read_archive() (skips corrupt/non-JSON lines), list_archive(limit) newest-first, archive_stats() (path/records/bytes/task_types/oldest_ts/newest_ts), prune_archive(retain_days) fail-closed atomic tmp+replace, records without ts treated oldest. Default retention now MemoryConfig.archive_retention_days.

CONFIG: SLO_MEMORY_ARCHIVE_RETENTION_DAYS env (default 30) added to MemoryConfig; prune_archive() defaults to it.

CLI: 'memory archive' command (--limit/-n, --prune-days) + module docstring/group help updated. Exports list_archive/archive_stats/prune_archive in domains/memory/__init__.py.

TESTS: TestArchive x9 in test_task_memory.py (newest-first, empty, stats, prune old/keep recent, keep-all-within-window, zero-window, no-file, corrupt-line); TestArchive x4 in test_memory_commands.py (stats+records, empty, prune confirmed/declined). Fixed a NameError from a bad edit that dropped 'stats = archive_stats()'.

DOCS: docs/ENVIRONMENT.md SLO_MEMORY_ARCHIVE_RETENTION_DAYS section; docs/integration/CLI_README.md added consolidate + archive rows.

VERIFIED: 153 tests pass (task_memory+cli memory+memory_service+maintenance+consolidation+startup_routers); py_compile clean; pycache cleared. Live CLI 'memory archive' shows real store. E2E smoke in tmp store: store task -> 1 archive record -> stats -> prune(0) removes 1.
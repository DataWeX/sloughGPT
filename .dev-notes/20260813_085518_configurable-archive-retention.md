---
id: 20260813_085518_configurable-archive-retention
title: Configurable archive retention
status: done
tags: memory,provenance,maintenance
created: 2026-08-13T08:55:18.455493+00:00
---

Configurable archive retention

Configurable archive retention (config-driven prune), layered core -> service -> API -> UI.

Core: MemoryConfig.set_archive_retention_days(days) thread-safe (clamps >= 0) + snapshot() returning all 8 runtime settings. Service: set_archive_retention() + config_snapshot() passthroughs.
API: ConfigRequest widened to optional enabled + archive_retention_days; POST /memory/config applies only provided fields and returns full snapshot; new GET /memory/config returns snapshot (enabled, min_chars, max_facts, store_path, sync_remember, consolidation_threshold, maintenance_interval_minutes, archive_retention_days). set_config return shape changed from {enabled} to full snapshot (backward-compatible: enabled key preserved).
Frontend: memory-controller getConfig() GET + updateConfig(partial) POST; MemoryConfigResult expanded to full snapshot. MemoryCard: 'Archive retention' row in Maintenance (number input 0-3650 + Save, fetches current value on mount, validates empty input, clamps), and Prune old now passes the configured retentionDays to archivePrune.
Tests: +6 core (set_archive_retention/clamp/zero/singleton/snapshot + mutation), +3 router (GET snapshot, POST retention, clamp), +5 controller (getConfig, updateConfig), +4 MemoryCard (loads value, saves, prune uses window, rejects empty). 33/33 MemoryCard, 70 knowledge+controller suite, 79 core memory tests, 45 maintenance+task_memory, 14 router all pass; tsc + eslint clean. docs/routers.md updated.
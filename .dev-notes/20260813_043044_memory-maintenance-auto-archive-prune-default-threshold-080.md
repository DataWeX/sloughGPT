---
id: 20260813_043044_memory-maintenance-auto-archive-prune-default-threshold-080
title: Memory maintenance: auto archive prune + default threshold 0.80
status: done
tags: memory,maintenance,cli
created: 2026-08-13T04:30:44.399887+00:00
---

Memory maintenance: auto archive prune + default threshold 0.80

Memory lifecycle completion: (1) maintenance_tick now auto-prunes the facts.jsonl provenance archive (default 30d via SLO_MEMORY_ARCHIVE_RETENTION_DAYS) before enqueueing consolidation; (2) default consolidation threshold corrected 0.85->0.80 (measured: identical 1.0, near-verbatim 0.845, paraphrase 0.586) so the default merges near-duplicate facts; (3) fixed KnowledgeMemory.clear_all persistence bug — _save_entries skipped writing empty stores, so a cleared store reloaded stale facts after restart; now persists [] when the file exists. Real data store was polluted by live CLI checks (2 test facts, gitignored) and has been cleaned + backed up to /tmp/opencode. 194 tests pass.
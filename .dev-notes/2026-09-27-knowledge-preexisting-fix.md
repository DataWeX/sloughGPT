---
title: Fix pre-existing knowledge router failures
status: done
date: 2026-09-27
---

# Knowledge router pre-existing failures (card 063)

## Failures
- `test_search_files` — 400 E_BAD_REQUEST "Path not found: routers"
- `test_stats_with_many_facts_terminates` — hang under 300s timeout

## Root causes + fixes
1. Engine bug: `extensions=["py"]` (undotted) never matched — index matches
   on dotted suffix. Fix: normalize to `"." + e.lstrip(".")` in
   `domain/knowledge/engine.py`.
2. Engine bug: `indexed_files: len(indexed)` returned dict key-count, not
   files. Fix: `int(indexed.get("files_indexed", 0))`.
3. Test bug: `path: "routers"` pointed nowhere. Fix: tmp_path fixture with
   real sample file + `assert len(results) >= 1`.
4. Timeout bug: 250 sequential POSTs each rebuild the dedup index (~210s
   total). Fix: one `/knowledge/bulk-ingest` per topic (2×125),
   `dedup_threshold=1.0`, disjoint text ranges so exact-dup skip logic
   (`score >= threshold`) doesn't drop distinct facts.

## Result
Commits `a570a4a84` (knowledge) + `ed23d43e1` (data_import, card 066),
merged to main as `254b59096` (merged in sloughgpt-router-wiring — main had
moved there; fix-recovery no longer holds main and its staged ruff batch from
the parallel session blocked merges there).
Post-merge gates on main: knowledge + data_import + lora_config **420/420**,
facade guard 2/2, ruff clean.
Gates: knowledge suite **19/19**, ruff clean. Reproduced-failures were
confirmed pre-existing via stash-compare (card 062 log).

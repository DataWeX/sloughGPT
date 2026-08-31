---
id: 20260813_045916_memory-archive-browser-ui
title: Memory archive browser UI
status: done
tags: knowledge,memory,frontend
created: 2026-08-13T04:59:16.941355+00:00
---

Memory archive browser UI

MemoryCard: 'View records' button in Maintenance opens a Provenance archive browser dialog (backed by /memory/archive) - last 20 records, per-task-type badges (remember/store/consolidate/other), human-readable summaries, timestamps, live archiveStats line, skeleton, EmptyCard, in-dialog Refresh. Algorithmic summary/badge helpers. Then Backup memory row: Export (list(1000) -> memory-export-<date>.json) and Import (parseMemoryImport: JSON array of strings/objects, CSV content/topic header, plain lines with optional [topic]; batched stores with progress line, reports stored/total). Exported parseMemoryImport + MemoryImportEntry.

This session: provenance browser deepened - record rows are now expandable buttons (chevron toggle) revealing the raw JSON payload (expandedRecordId state), plus an Export button in the dialog footer (archive(1000) -> memory-archive-<date>.json, disabled when no records). Tests: +2 (expand/collapse raw payload with URL field, archive export download). 29/29 MemoryCard tests, 39/39 knowledge suite pass; tsc + eslint clean.
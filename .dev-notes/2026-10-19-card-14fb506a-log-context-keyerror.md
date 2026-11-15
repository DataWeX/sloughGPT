---
id: 14fb506a
title: "PROD BUG: explicit request_id in StructuredLogger extra crashes with KeyError whenever a request/correlation context is active"
status: review
board_id: Main
date: 2026-10-19
branch: fix/logkeyerr
tags: [bug, logging, inference]
---

# Card 14fb506a — make the record-factory KeyError structurally impossible

Full card id: `14fb506a-96c0-4c61-a153-c244a77f8c66` (board column: in_progress).

## LANDED 2026-10-19 (Gene batch of 3, done inline — subagent nesting blocked at depth 1)

- **Commit**: `006fb562d` on branch `fix/logkeyerr` (worktree `.wt-logkeyerr`,
  cut from HEAD 681bca18e). Pathspec commit, 3 files, 225 insertions /
  78 deletions — `git show --stat` matches the edit set exactly, no foreign hunks.
- **Fix (foreman option 1)**: `_enriched_record_factory` (in
  `domain/logging/_internal/config.py` — NOT apps/api/server/...config.py; the
  brief's path predates the domain restructure) is REMOVED; enrichment now
  lives in a stateless `ContextEnrichmentFilter` attached to the console,
  file and OutputBuffer-bridge handlers (shared instance). Explicit extras
  merge first; correlation/log-context only fills gaps — collision
  impossible for any key. Hand-made handlers opt in via the same filter.
- **Path adaptation vs brief**: queue-full site is
  `domain/infrastructure/_internal/request_queue.py` (`PriorityRequestQueue.acquire`,
  same StructuredLogger warning at line ~105); no queue code needed changing —
  the defect was logging-side.
- **Fail-before proven** (log: `~/.cache/slog-gates/gene-batch-20261019/A-logging/repro-before.txt`):
  `KeyError: "Attempt to overwrite 'request_id' in LogRecord"` on both
  regression tests, incl. the queue-full path leaking KeyError from
  `logger.warning` instead of raising RuntimeError. Pass-after: 3/3
  (`repro-after.txt`).
- **Trap audit**: caplog usage in the logging tests is limited to one
  suppression-summary test (no request_id assertion) — unaffected.
  `test_correlation_id_in_stdlib_log` hand-attaches its own root handler;
  updated to opt into `ContextEnrichmentFilter` with an explanatory comment
  (assertion preserved). BridgeHandler tests: zero correlation dependencies.
  `TestRecordFactory` rewritten as `TestContextEnrichmentFilter` (4 tests,
  incl. explicit-wins precedence).
- **Gates**: full logging suite 674 passed (10 files); queue suites 43
  passed + 2 FAILED pre-existing (`TestMetrics::test_depth_per_priority`,
  `TestEdgeCases::test_depth_api` — proven at pristine HEAD 681bca18e in the
  untouched `.wt-astland` tree; root cause: depth-counter accounting in
  PriorityRequestQueue, unrelated to logging); ruff check + format clean.
- Status: review → done when foreman re-verifies and the branch lands.

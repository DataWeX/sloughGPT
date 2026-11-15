---
id: 20260929_086
board: 8182563a-b55f-43e5-8384-7eab88698365
title: "Card 086 — Logs unification: fold monitor into logs + beautiful colored output"
status: done
date: 2026-09-29
tags: [cli, logs]
---

# Logs unification: fold monitor into logs + beautiful colored output

## Background

User rules from 2026-08-27/28: 'don't build a separate command for monitor, integrate it into the log' + 'make the log output for all more beautiful, it's a selling point' + punchy info for ALL processes, not just metrics/stats. Today apps/cli has separate commands/monitor.py AND commands/logs.py (--dashboard partial). Fold in, complete color coding, de-clutter live dashboard.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Evidence

Landed 2026-09-29 on `docs/api-doc-gap-audit` branch/worktree `.wt-audit`
(uncommitted; file sets disjoint from cards 083/088/090).

**Discovered while folding — two pre-existing total breakages on main:**
1. `sloughgpt logs` (and any wrapped command) crashed at dispatch:
   `AttributeError: '_ClickCommandWrapper' object has no attribute 'func'`
   (`core/framework.py run_command` dispatches via `cmd.func`; the wrapper
   duck-typed Command but never exposed `func`).
2. `sloughgpt train monitor` crashed: framework `Context.invoke` forwarded
   kwargs into the raw click command object → `TypeError: Context.__init__()
   got an unexpected keyword argument`.

**Changes:**
- `core/framework.py Context.invoke` — raw click commands now get defaults
  filled via `make_context(..., resilient_parsing=True)` + `ctx.params`, then
  overrides applied, and the real `callback` is called. Fixes the class of
  bug, not just train monitor.
- `cli.py _ClickCommandWrapper` — `self.func = self.__call__` (completes the
  Command duck-type) + `hidden` constructor param.
- **Fold:** `commands/monitor.py` deleted (333 lines). One implementation
  lives in `commands/logs.py`; `monitor` remains as a 30-line deprecated
  click alias that forwards to `logs.callback(dashboard=True, ...)` —
  registered **hidden** (help shows only `logs`), `monitor --help` explains
  the move, muscle memory keeps working.
- `groups/train.py train_monitor` — delegates to logs --dashboard via the
  fixed `ctx.invoke`; `ctx.obj.get(...)` defaults (was bare `["host"]`).
- `routers/dashboard.py` docstring — `sloughgpt logs --dashboard`.

**Beautiful output (`_format_line`):** level pills — DEBUG dim cyan,
INFO green, WARNING bold yellow, ERROR white-on-red bg, CRITICAL
white-on-magenta bg, all padded inside the pill to 8 cols so streams stay
aligned; tag now cyan; ERROR/CRITICAL messages tinted; plain mode keeps the
8-wide level column and zero ANSI.

**Punchy processes:** full dashboard rows now carry a colored type badge
(training/self-train/auto-train/download) beside label+detail+progress;
**compact mode no longer drops processes** — shows
`PROC  2 active  1× training  1× download` (counts by type) instead of
metrics-only.

## Gates

- `apps/cli/tests`: **437 passed / 1 skipped** (baseline 422 + 15 new
  `test_logs_monitor_fold.py`)
- server suite: **1209 passed / 0 failed**
- ruff (all touched files): clean
- manual: `logs --file` exits 0 with formatted line; `monitor --json` runs;
  `train monitor` renders dashboard retry loop; top-level help lists `logs`,
  not `monitor`; `monitor --help` shows deprecation notice

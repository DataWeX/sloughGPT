---
id: 20260927_067_cmddev_startup_int
status: done
title: cmd_dev startup SIGINT phase — verify early-INT ends session + cleanup (journey A phase 0)
tags: [cli, startup, signal, journey, dev-server]
---
# cmd_dev startup SIGINT (journey A phase 0)

## Problem
Landed 3e1d71da8 registered cmd_dev's flag-only handler at function entry,
but no E2E proves what happens when SIGINT fires DURING startup (pre-dashboard):
does the session end, and do spawned children (uvicorn + web child) get cleaned
up? Flag-only handler does no cleanup itself — depends on the startup loops
honoring `shutdown` and the `finally` running.

## Plan
1. Read cmd_dev startup wait: who notices `shutdown` pre-dashboard, does finally run.
2. Fix gaps if any (real bug → cleanup at entry handler or loop guards).
3. journey_api phase 0: INT at t+2 → exit <=10s + no orphan uvicorn/web + ports free.
4. Gates: ruff + cli suite + journey A. Commit fast (wipes!), land.

<!-- claimed by opencode 2026-09-27 -->

<!-- completed by opencode 2026-09-27 -->

## Result (2026-09-27) — done (commit 767cadc04, main=767cadc04)

1. No bug found: both handler paths funnel to `finally` — flag handler
   (entry, pre-dashboard) and DevDashboard.serve's swap-in handler
   (apps/cli/src/core/tui.py:1149) set their flag → loop exits →
   `_cleanup` + `_print_summary` ("Dev Server Stopped").
2. journey A phase 0: INT after cli+uvicorn visible + 2s → exit 5s,
   summary printed, no orphan (cli/uvicorn/npm), ports 8177+4569 freed.
   Journey A now 11/11 (was 6/6).
3. M6 mutation (registration moved back to pre-dashboard): caught by
   registration-before-spawn unit source-order test (reg 6640 > spawn 1717).
4. Gates: unit 415+1skip, journey A 11/11, bash -n. Landed ff in recovery.
5. Method change: worktree moved IN-TREE (.wt-sig) — survived interruptions
   that killed 10 /tmp/opencode worktrees. Worktree removed post-land.

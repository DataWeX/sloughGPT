---
title: TUI LiveDisplay + Linux Metrics Fixes
date: 2026-08-31
tags: [tui, cli, bugfix, linux]
status: done
---

# TUI LiveDisplay + Linux Metrics Fixes

## Summary
Three fixes for the CLI dev server TUI and startup logic.

## Changes

### 1. LiveDisplay.update() — zero-newline rewrite
- **File:** `apps/cli/src/core/tui.py` (line 739)
- **Problem:** `\n`-delimited writes caused terminal scroll/drift during animation
- **Fix:** Replaced with `\033[row;1H` cursor positioning — zero newlines written during updates
- **Verified:** Automated test confirms zero `\n` in output

### 2. Linux metrics collection
- **File:** `apps/cli/src/core/tui.py` (line 912)
- **Problem:** `_collect_metrics()` used macOS-only commands (`top -l`, `vm_stat`), silently failed on Linux showing 0.0
- **Fix:** Added `/proc/stat` (CPU) and `/proc/meminfo` (memory) for Linux; macOS path preserved

### 3. Web restart cwd bug
- **File:** `apps/cli/src/commands/dev.py` (line 978)
- **Problem:** Monitor loop restart used `standalone_dir` as cwd, but `server_js` could be at `standalone_dir/apps/web/server.js`
- **Fix:** Changed to `server_js.parent` to match initial start logic

## Verification
- `python3 scripts/terminal-tui.py 1` — single-line spinner, zero drift
- `python3 scripts/terminal-tui.py 2` — multi-line fixed block
- `python3 scripts/terminal-tui.py 3` — full dashboard with live metrics

## Result (2026-09-26)
All three fixes verified present on main:
1. LiveDisplay zero-newline rewrite — `\033[{i};1H{line}\033[K` cursor
   positioning in `apps/cli/src/core/tui.py` (update path writes no `\n`).
2. Linux metrics — `/proc/stat` (tui.py ~957) and `/proc/meminfo` (~976);
   macOS path preserved.
3. Web restart cwd — `apps/cli/src/commands/dev.py` resolves
   `server_js` candidates and launches with
   `cwd=str(server_js.parent.resolve())` (line ~1178) + fallback to
   `npm run dev` when standalone build absent.
Card was stale (fixes had landed but status never updated); closed as-is,
no code changes needed.

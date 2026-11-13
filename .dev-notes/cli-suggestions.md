---
title: CLI Post-Command Suggestions
date: 2026-08-31
tags: [cli, qol, suggestions]
status: done
---

# CLI Post-Command Suggestions

## What was built
- Post-command suggestions shown after `model list`, `model status`, `train start`, etc.
- Usage tracking in `~/.config/sloughgpt/usage_stats.json`
- Auto-correct still works: `md` → `model`

## How it works
- `cli.py:main()` parses `sys.argv` for command path before Click runs
- After `cli()` returns (catching `SystemExit`), checks `_SUGGESTIONS` dict
- Only shows in TTY mode
- `_SUGGESTIONS` dict in `cli_group.py` maps command → suggested follow-up

## Key files
- `apps/cli/src/cli.py:3594` — main() with suggestion hook
- `apps/cli/src/core/cli_group.py:39` — _SUGGESTIONS dict

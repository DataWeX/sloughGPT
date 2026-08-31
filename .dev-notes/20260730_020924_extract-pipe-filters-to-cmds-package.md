---
id: 20260730_020924_extract-pipe-filters-to-cmds-package
title: Extract pipe filters to cmds/ package
status: done
tags: shell,refactor
created: 2026-07-30T02:09:24.345301+00:00
---

Extract pipe filters to cmds/ package

Extracted 8 pipe-filter commands (grep, head, tail, wc, tee, sort, uniq, less) from repl.py into cmds/filters.py. Each filter reads piped input from env['_piped_input'], writes output via io.print(). Removed _cmd_* methods, COMMANDS dict entries, and help text from repl.py. All 200+ shell tests pass. 70 _cmd_* methods remain in repl.py (~30 extracted total so far).
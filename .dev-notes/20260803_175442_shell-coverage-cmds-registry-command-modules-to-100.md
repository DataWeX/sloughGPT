---
id: 20260803_175442_shell-coverage-cmds-registry-command-modules-to-100
title: Shell coverage: cmds/ registry + command modules to 100%
status: done
tags: shell,coverage,cmds
created: 2026-08-03T17:54:42.050967+00:00
---

Shell coverage: cmds/ registry + command modules to 100%

Wave W: all 5 cmds modules at 100% (284 stmts, 0 missed). New tests/test_shell_cmds.py (49 tests) covering CmdModule lazy-load/discover, health, models_cmd, souls_cmd, data_cmds via FakeConsole (records output, no spinner threads) + FakeApi stub. Dead code removed: _dict_val (0 callers) in data_cmds.py; __init__ guard in discover() (iter_modules never yields __init__). Verified: 100% coverage with test_shell_repl, full shell suite green (repl/commands/subsystems/tui/integration/lifecycle), py_compile OK.
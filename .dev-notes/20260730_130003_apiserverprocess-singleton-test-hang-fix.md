---
id: 20260730_130003_apiserverprocess-singleton-test-hang-fix
title: APIServerProcess singleton + test hang fix
status: done
tags: shell,runtime,tests
created: 2026-07-30T13:00:03.817664+00:00
---

APIServerProcess singleton + test hang fix

Root cause: ShellREPL.run() calls api.start() which spawns a subprocess and waits up to 120s. Each DaitRuntime creates a fresh APIServerProcess, so no two instances shared a process reference.

Fix:
- runtime.py: Replaced instance-level _proc/_lock/_started_at with module-level _shared_proc/_shared_lock/_shared_started_at. Only the first start() call spawns a process; subsequent calls see _shared_proc is not None and return immediately.
- test_shell_repl.py: Added patch('...APIServerProcess.start') to the repl fixture so run() never spawns a process in unit tests.

Result: 298 shell tests pass in 2.07s (was hanging at 90s+ timeout).
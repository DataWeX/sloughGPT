---
id: 20260813_045601_cli-memory-enable-disable-commands
title: CLI memory enable-disable commands
status: done
tags: cli,memory
created: 2026-08-13T04:56:01.462590+00:00
---

CLI memory enable-disable commands

Added sloughgpt memory enable/disable subcommands to close the CLI gap for the memory master switch (previously only settable via API /memory/config and the UI toggle).

- apps/cli/src/commands/memory.py: cmd_memory_enable(args) calls MemoryService.set_enabled(enabled), prints confirmation
- apps/cli/src/cli.py: memory enable / memory disable click commands wired after stats; comment header updated
- apps/cli/tests/test_memory_commands.py: TestEnable class (2 tests: enable sets True, disable sets False, both assert set_enabled called and status text)
- docs/integration/CLI_README.md: memory command table + example updated with enable/disable

Verification: CLI suite 113 passed (was 111); py_compile clean; live check shows 'Memory disabled' / 'Memory enabled' state reflected in memory stats. No API change needed - CLI calls service directly like other memory commands.
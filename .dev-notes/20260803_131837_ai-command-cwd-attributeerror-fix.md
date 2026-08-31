---
id: 20260803_131837_ai-command-cwd-attributeerror-fix
title: ai command cwd AttributeError fix
status: done
tags: shell,bugfix,ai
created: 2026-08-03T13:18:37.808236+00:00
---

ai command cwd AttributeError fix

Bug: `ai <query>` in the TUI shell crashed with `'DaitRuntime' object has no attribute 'cwd'` at repl.py:4460 (`ctx_parts = [f"  Current directory: {self.os.cwd}"]`).

Fix: DaitRuntime has no `cwd`; the shell tracks the working directory via `os.getcwd()` everywhere else (repl.py:508, 1967, 1970, 1984). Replaced `self.os.cwd` with `os.getcwd()`.

Tests added (tests/test_shell_repl.py::TestAiCommand):
- test_ai_query_uses_real_cwd: patches _probe_api available=True + _spinner_call, runs _cmd_ai, asserts generated command echo (exercises the exact crash line).
- test_ai_falls_back_to_keyword_match_when_api_down: api down path.

Verified: TestAiCommand 2/2, full shell suites (repl + tui_repl + tui_live) 378 passed.
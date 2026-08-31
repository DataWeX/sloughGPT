---
id: 20260801_063200_unify-to-single-curses-tui-drop-packagestui-ts
title: Unify to single curses TUI; drop packages/tui-ts
status: done
tags: tui,cli,shell
created: 2026-08-01T06:32:00.030276+00:00
---

Unify to single curses TUI; drop packages/tui-ts

Unified the repo to a single TUI and pushed.

Decision: sloughgpt tui and sloughgpt shell --tui both boot the in-core split-pane curses TUI (TuiRepl, packages/core-py/domains/shell/). It runs in-process (no HTTP), shows infra log console + shell output + chat output, and is the only TUI that works without a running API server. The TypeScript Ink TUI (packages/tui-ts) was a thin HTTP client to the FastAPI server (only client.health() wired); it duplicated terminal UI with no in-process value, so it was removed.

Changes (commit f0d2ea0):
- Deleted packages/tui-ts/ entirely (package.json, src/App.tsx, src/cli.tsx, parseBaseUrl, tests, config).
- Updated references: apps/README.md, docs/STRUCTURE.md (incl. stale test-tui-ts CI line), .dockerignore, SPEC.md, .agents/skills/SKILL.md, scripts/build.sh (removed sloughgpt-tui build).
- Kept packages/sdk-ts/typescript-sdk (standalone published TS SDK, independent of tui-ts).
- Line-mode REPL preserved as default: sloughgpt shell (no --tui). Verified: sloughgpt shell -c 'echo hi' -> 'hi' (offline, exit 0).

Verification: TuiRepl imports cleanly; shell test suites (test_shell_repl, test_shell_tui_repl, test_shell_pane, test_shell_surface, test_shell_log_buffer, test_shell_logger) all pass.

Note: earlier commit b024103 (tui: drop Python apps/tui) accidentally included apps/cli/src/commands/train.py (_softmax_np); content is correct, only the commit grouping is off, and it is already pushed, so left as-is.
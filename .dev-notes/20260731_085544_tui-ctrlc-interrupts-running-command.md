---
id: 20260731_085544_tui-ctrlc-interrupts-running-command
title: TUI Ctrl+C interrupts running command
status: done
tags: shell,tui,cli
created: 2026-07-31T08:55:44.216223+00:00
---

TUI Ctrl+C interrupts running command

TuiRepl._interrupt_active() raises KeyboardInterrupt in the command thread via PyThreadState_SetAsyncExc (guarded, best-effort). Ctrl+C now interrupts a running command (repl's _dispatch prints 'Aborted'); a second Ctrl+C exits. 4 new tests; 35 TUI + 267 repl + 492 shell/logging pass. PTY probe: watch 0.2 echo tick ran 3 iterations, Ctrl+C -> Stopped, TUI alive, follow-up date ran. SHELL.md + ROADMAP updated.
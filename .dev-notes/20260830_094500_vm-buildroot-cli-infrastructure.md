---
title: VM + Buildroot CLI Infrastructure Complete
created: 2026-08-30T09:45:00.000000+00:00
updated: 2026-08-30T09:45:00.000000+00:00
tags: vm, buildroot, cli, shell, systems, infrastructure
status: done
---

## Session Summary

Completed VM and Buildroot CLI infrastructure for SloughGPT. All 466 tests passing.

## What Was Built

### VM CLI Commands (`apps/cli/src/commands/vm.py`)
- `vm status` — Shows VM info, syscalls, RBAC roles
- `vm list` — Lists 16 available assembly programs
- `vm run --file prog.asm` — Executes assembly, shows registers/output
- `vm info` — Detailed memory/CPU/RBAC info
- `vm debug` — Interactive debugger with breakpoints, step-through, memory inspection

### Buildroot CLI Commands (`apps/cli/src/commands/build.py`)
- `build status` — Buildroot initialization and image status
- `build run` — Build custom Linux image via Docker
- `build clean` — Clean build output
- `build install` — Install image to web public directory

### Buildroot Infrastructure (`buildroot/`)
- `configs/sloughgpt_defconfig` — x86_64 Buildroot defconfig
- `configs/linux-soughgpt.defconfig` — Linux kernel config (VT, networking, modules)
- `configs/busybox-soughgpt.defconfig` — BusyBox config (ash shell, core utils)
- `overlays/etc/init.d/S00setup` — Mount proc/sys/devpts/tmp
- `overlays/etc/init.d/S99dait` — Dait shell init (runlevels 0-6)
- `overlays/etc/fstab` — Mount table
- `overlays/etc/profile` — Shell environment
- `overlays/opt/sloughgpt/etc/dait.conf` — Dait configuration
- `packages/sloughgpt/sloughgpt.mk` — Package makefile
- `packages/sloughgpt/Config.in` — Kconfig entry
- `post-build.sh` — Rootfs customization hook
- `post-image.sh` — Image creation hook
- `Dockerfile` — Reproducible build environment (Ubuntu 22.04)
- `build.sh` — Docker-based build script
- `README.md` — Build instructions

### VM Debugger (`packages/core-py/domains/shell/vm_debugger.py`)
- `Debugger` class — Interactive debugging
- `SymbolTable` — Symbol resolution (name → address)
- Breakpoints with labels, hit counts
- Watchpoints for memory monitoring
- Step-over, step-out, continue execution
- Memory hex dump, register dump, flag dump
- Stack trace, trace analysis

### Kernel Module Loader (`packages/core-py/domains/shell/addons/module_loader.py`)
- `ModuleLoader` class — Runtime addon management
- Dynamic addon discovery from directories
- Load/unload/reload with dependency tracking
- Hot-reload support for development
- Error isolation per module

### Systems Engineering Agent
- `.opencode/agents/systems-engineer.md` — Agent definition
- `.opencode/skills/systems-engineering/SKILL.md` — Skill with kernel, VM, Buildroot knowledge
- `.opencode/commands/systems.md` — Command binding
- `opencode.json` — Registered agent + command

### CLI Integration (`apps/cli/src/cli.py`)
- Registered `vm` group with 5 commands
- Registered `build` group with 4 commands
- Uses `_ns()` helper for SimpleNamespace args
- All commands use `getattr(args, ...)` for attribute access

### Test Coverage
- `test_vm_debugger.py` — 26 tests (debugger, symbol table, module loader)
- `test_cli_commands.py` — 28 tests (CLI files, buildroot, Dait init)
- `test_buildroot.py` — 35 tests (configs, overlays, packages, v86)
- Total: 466 tests passing across all shell test suites

## Files Modified
- `apps/cli/src/commands/vm.py` — Rewrote with getattr(), fixed list command
- `apps/cli/src/commands/build.py` — Already correct
- `apps/cli/src/cli.py` — Added vm + build Click groups
- `apps/cli/src/commands/__init__.py` — Updated docstring
- `.opencode/agents/systems-engineer.md` — Added debugger/module loader docs
- `.opencode/skills/systems-engineering/SKILL.md` — Added capabilities

## Known Issues
- `vm run "inline assembly"` has shell escaping issues for newlines — use `--file` instead
- `run()` without `max_steps` can hit memory bounds on programs that don't halt — CLI uses `max_steps=100000`
- `build status` shows truncated output in pipe due to terminal width

## Next Steps
- Fix the `run()` method in vm_engine.py to properly detect HLT with 0x66 prefix
- Add `vm run <program_name>` shortcut to run built-in programs by name
- Add `build init` command to clone Buildroot repo
- Create integration tests for CLI commands end-to-end

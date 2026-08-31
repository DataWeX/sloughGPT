---
id: 20260809_152850_vm-console-persist-role-max-steps-clamp-steps-input
title: VM console: persist role + max-steps, clamp steps input
status: done
tags: vm,console,frontend
created: 2026-08-09T15:28:50.567953+00:00
---

VM console: persist role + max-steps, clamp steps input

Persisted role (vm-role) and maxSteps (vm-max-steps) via localStorage, load-on-mount with validation (role in {user,admin,kernel}, steps clamped to [1,1_000_000], fallback DEFAULT_MAX_STEPS). clampSteps() applied at run time and in the step-limit banner. +4 page tests (46 total), +1 e2e spec (10 total), docs/VM_CONSOLE.md Persistence section. tsc exit 0; full suite 327 files / 3162 tests pass.
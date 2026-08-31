---
id: 20260809_093011_vm-console-role-selector-train-sample-program
title: VM console role selector + train sample program
status: done
tags: vm,web,training
created: 2026-08-09T09:30:11.527164+00:00
---

VM console role selector + train sample program

Completed the VM training syscall surface: the Training card could never activate from the UI because /vm/run always ran as USER and SYS_TRAIN_START requires Permission.TRAINING (ADMIN role).

Changes:
- app/(app)/vm/page.tsx: added role selector (user/admin/kernel) in the top bar (after keyboard input, additive), wired role into handleRun via vmController.run(source, { role, ... }), deps [source, maxSteps, role, debug].
- Added 'train' sample program to DEFAULT_PROGRAMS — calls SYS_TRAIN_START (EAX=28, EBX=config JSON '{"dataset":"shakespeare","epochs":1}') then HLT; the Training card polls the job to completion. Verified it assembles (50 bytes) with the real X86Assembler.
- Denied syscalls surface as EAX=-2 (0xFFFFFFFE) from _check_perm.

Tests (page.test.tsx): added cleanup() in afterEach (RTL auto-cleanup does NOT run — vitest globals disabled, so DOM accumulated across tests and role state leaked); 5 new tests (train button, train source load, role selector default, role passed to run, role default user) scoped via within(container). 25/25 pass.

Also fixed pre-existing SoulsPage test failure: mockListWeightSnapshots was never given mockResolvedValue([]) so the page's listWeightSnapshots().catch() threw on undefined — added to beforeEach. 5/5 pass.

Verification: npx tsc --noEmit exit 0; full web suite 325 files / 3125 tests all pass (3 errors are intentional error-path stderr logging); targeted vm tests 32/32.

Next free lane: vm RBAC role enforcement tests (SYS_TRAIN denied at USER returns -2 in EAX), mobile, security, knowledge-enrichment.
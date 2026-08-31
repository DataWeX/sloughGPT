---
id: 20260809_112747_vm-console-permission-denied-hint
title: VM console permission-denied hint
status: done
tags: vm,frontend
created: 2026-08-09T11:27:47.542952+00:00
---

VM console permission-denied hint

Added a role-aware denial banner to the VM console: when a run finishes with EAX=0xFFFFFFFE (-2), the Result card now shows a friendly warning explaining a syscall was denied for the current role and directing the user to switch to the admin role. Derived flag permissionDenied checks EAX hex case-insensitively. 2 new tests (denial shown, normal run hidden). Verified: vm page 27/27, vm-controller 7/7, tsc exit 0.
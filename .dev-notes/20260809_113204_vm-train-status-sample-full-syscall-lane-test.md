---
id: 20260809_113204_vm-train-status-sample-full-syscall-lane-test
title: VM train-status sample + full syscall lane test
status: done
tags: vm,training,frontend
created: 2026-08-09T11:32:04.071231+00:00
---

VM train-status sample + full syscall lane test

Added 'train-status' sample program to the VM console DEFAULT_PROGRAMS exercising the full training syscall surface: SYS_TRAIN_STATUS (eax=29) then SYS_TRAIN_GET_RESULT (eax=30, ECX=buf 0x90000, EDX=size), storing both results to guest memory. Verified assembles with real X86Assembler (47 bytes). Core: new test_train_full_flow_start_status_result runs START->STATUS->GET_RESULT under ADMIN with a fake bridge and asserts job_id=1, status code 1 (completed), 13 bytes written, and result JSON '{"loss": 1.5}' read back. Frontend: 2 new tests (button renders + source loads). Verified: core VM cluster 1851 pass, vm page 29/29, tsc exit 0.
---
id: 20260809_115755_vm-console-docs-reference-panel-training-syscalls-builtins-p
title: VM console docs + reference panel training syscalls + builtins parity
status: done
tags: vm,docs,frontend
created: 2026-08-09T11:57:55.918385+00:00
---

VM console docs + reference panel training syscalls + builtins parity

Created docs/VM_CONSOLE.md documenting the /vm console: architecture, endpoints, RBAC model (USER/ADMIN/KERNEL roles, EAX=-2 denial), the training syscall surface (SYS_TRAIN_START/STATUS/GET_RESULT args and returns), sample program table, and relevant files. Added a VM router row to docs/API.md pointing at the new doc. Reference panel Interrupts cell now lists EAX=28/29/30 training syscalls (1 new test). /vm/builtins catalog now includes 'train-status' (test assertion added). Verified: vm page 30/30, vm router 14/14, tsc exit 0.
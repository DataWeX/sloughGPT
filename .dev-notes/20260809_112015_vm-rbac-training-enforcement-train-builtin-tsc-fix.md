---
id: 20260809_112015_vm-rbac-training-enforcement-train-builtin-tsc-fix
title: VM RBAC training enforcement + train builtin + tsc fix
status: done
tags: vm,rbac,training,frontend
created: 2026-08-09T11:20:15.634071+00:00
---

VM RBAC training enforcement + train builtin + tsc fix

Added RBAC enforcement tests for SYS_TRAIN syscalls. Core-py (test_vm_rbac.py): USER gets EAX=0xFFFFFFFE (-2) on SYS_TRAIN_START/STATUS/GET_RESULT and the training bridge is never invoked; ADMIN returns job_id 1 and reaches the bridge. API (test_vm_router.py): USER-role /vm/run of the train program returns training_job_id null + EAX 0xFFFFFFFE in registers with no bridge call; /vm/builtins now lists 'train'. Fixed tsc error at models/page.tsx:213 by adding request_count to HealthStatus. Verified: core VM cluster 1850 pass, vm router 14 pass, full web suite 378 files/3781 tests pass, tsc exit 0.
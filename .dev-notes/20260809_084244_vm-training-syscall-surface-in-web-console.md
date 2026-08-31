---
id: 20260809_084244_vm-training-syscall-surface-in-web-console
title: VM training syscall surface in web console
status: done
tags: vm,training,frontend,backend
created: 2026-08-09T08:42:44.732983+00:00
---

VM training syscall surface in web console

Backend: /vm/run captures SYS_TRAIN_START job_id (VMRunResponse.training_job_id); new GET /vm/training/jobs/{id} endpoint (VMTrainingJobResponse) delegating to VMTrainingBridge.job_info(); 4 new router tests (13/13). Core: fixed latent spawn() stack-overflow fault at default 0x100000 memory (stack clamped into _mem_size); added bridge.job_info(). Frontend: vmController.trainingJob(id) + VMTrainingJob type; VM page Training card with 3s polling (terminal states stop polling, airtight cleanup); 4 page tests + 2 controller tests. Verified: 27/27 targeted vitest, 13/13 router pytest, 3128/3128 full web suite executed tests pass. Editor in-flight: useChatBookmarks deletion leaves 3 tsc errors; 50 chat test files deleted mid-suite-run (files-not-found, not failures).
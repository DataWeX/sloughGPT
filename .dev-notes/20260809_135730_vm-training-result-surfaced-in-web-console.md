---
id: 20260809_135730_vm-training-result-surfaced-in-web-console
title: VM training result surfaced in web console
status: done
tags: vm,rbac,web
created: 2026-08-09T13:57:30.319858+00:00
---

VM training result surfaced in web console

Third increment: completed the VM training lifecycle with job stop — the Training card can now cancel a running job via the bridge.

- packages/core-py/domains/shell/vm_training_bridge.py: VMTrainingBridge.stop(job_id) proxies POST /training/jobs/{api_job_id}/stop, marks local record 'stopping'; returns False on missing job / missing api_job_id / API error. +4 tests in test_vm_training_bridge_more.py.
- apps/api/server/routers/vm.py: POST /vm/training/jobs/{job_id}/stop endpoint (404 when bridge refuses). +2 router tests. 19/19 pass.
- apps/web/lib/vm-controller.ts: stopTrainingJob(jobId). +1 controller test. 10/10 pass.
- apps/web/app/(app)/vm/page.tsx: polling refactored into pollTrainingJob useCallback; handleStopTraining calls stop then re-polls; TrainingCard gets onStop prop + Stop button (ghost, sm) shown only for running/queued/starting jobs; train sample comment mentions Stop.
- apps/web/app/(app)/vm/page.test.tsx: +1 test (Stop button appears on running job, click calls stopTrainingJob(7)). 34/34 pass.
- apps/web/cypress/e2e/vm-page.cy.ts: +1 spec (run → Stop button → stop POST). 8 specs; mockVm unchanged.
- docs/VM_CONSOLE.md: /vm/training/jobs/{id}/stop row + samples section.
- Verified: tsc --noEmit exit 0; web 44/44 (page + controller); router 19/19; bridge+rBAC cluster 60/60; py_compile clean; pycache cleared.
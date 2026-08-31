---
id: 20260808_023433_models-422-device-regression-tests-locked-in
title: Models 422 device regression tests locked in
status: done
tags: server,tests,models,process-guard
created: 2026-08-08T02:34:33.941324+00:00
---

Models 422 device regression tests locked in

Added regression tests locking in the /models device-null 422 fix across three layers. (1) apps/api/server/tests/test_process_guard_controller.py: adopt_process_guard now asserts _current_device/_loaded_at are set, reads guard.device (mps), defaults to cpu, and preserves loaded_at across re-adopts. (2) tests/server/test_models_controller.py TestGetCurrentModel: model-without-device -> None, device-without-model -> None, both set -> dict with device, missing loaded_at -> None field. (3) tests/server/test_models_router.py: E2E regression using a REAL ModelsController with an adopted fake guard -> GET /models 200 with device 'cpu' (was 422/500 when the fix is reverted). Verified regression strength by reverting all three fix lines (guard, adopt device set, 'or cpu' fallback) -> new tests fail; restored -> 44/44 targeted pass. Full tests/server suite: 824 passed, 68 deselected (2 consecutive clean runs; earlier 3 inference no-model failures were pre-existing order/timing flake, pass in isolation and on re-run). Live server /models: HTTP 200 device 'cpu'.
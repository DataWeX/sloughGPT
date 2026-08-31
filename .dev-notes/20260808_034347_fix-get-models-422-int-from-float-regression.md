---
id: 20260808_034347_fix-get-models-422-int-from-float-regression
title: Fix GET /models 422 int_from_float regression
status: done
tags: models,validation,bugfix
created: 2026-08-08T03:43:47.798315+00:00
---

Fix GET /models 422 int_from_float regression

Fixed GET /models 422 int_from_float regression at two layers: (1) controllers/models.py:847-848 list_hf_models coerces num_parameters/vocab_size via int(); (2) routers/models.py list_models coerces parameters/vocab in loaded+available branches and passes coerced params to both ModelInfo and _describe_model. 3 regression tests in TestListModels (fractional float 1558000000.5, None, str -> all 200 ints). Fixed stale TestLoadModel::test_load_records_load_event_on_success (expected device=auto, route records resolved device=cpu from fixture) -> updated to device=cpu + added test_load_event_records_requested_device_when_result_has_none (fallback to req.device.value=auto). FULL SUITE: tests/server 994 passed/68 deselected, apps/api/server/tests 318 passed, combined models/mobile/process-guard 71 passed, live GET /models 200. HARDENING SWEEP: audited all strict-int response fields across schemas (datasets size_bytes/num_samples from st_size+line counts = safe ints; health inference_count and feedback thumbs_up/down/total/message_count all +=1 int counters = safe; LoadModelResponse.parameters never emitted by load_model = defaults to 0; /models/catalog, /hf, /current untyped = immune). Remaining: none — only models.py params/vocab were float/None-tainted. FRONTEND CLEANUP: apps/web/lib/query/hooks.ts:5 dead api.getModels() docstring (legacy api.ts deleted) -> updated examples to modelController.list()/soulsController.listCheckpoints()/modelController.load(); tsc exit 0, query tests 35/35.
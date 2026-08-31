---
id: 20260808_035359_model-load-event-records-resolved-device
title: Model load event records resolved device
status: done
tags: server,tests,models,device
created: 2026-08-08T03:53:59.554211+00:00
---

Model load event records resolved device

routers/models.py load_model previously recorded the requested device enum (device=cuda) into the model lifecycle event even when the controller validated and fell back to cpu. Now records result.get('device') (resolved) falling back to the requested value when absent. 2 new tests in tests/server/test_models_router.py::TestLoadModel: resolved-device recording (regression-proven, failed with device=cuda before fix) + fallback when response omits device. Live verified: POST /models/load device=cuda -> response device cpu, event detail 'device=cpu'. Full tests/server: 977 passed, 68 deselected (earlier 2 self_train + 5 multimodal failures were order-dependent pollution, both suites pass in isolation).
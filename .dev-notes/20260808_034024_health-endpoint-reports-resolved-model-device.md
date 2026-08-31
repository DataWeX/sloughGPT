---
id: 20260808_034024_health-endpoint-reports-resolved-model-device
title: Health endpoint reports resolved model device
status: done
tags: server,tests,health,device
created: 2026-08-08T03:40:24.692397+00:00
---

Health endpoint reports resolved model device

GET /health now returns the resolved device (cpu/mps/cuda). Added _get_model_device() in controllers/health.py resolving models controller first (authoritative resolved device after _resolve_device validation), then ModelRegistry default model, else None. Wired 'device' into get_basic_health result dict. Frontend HealthStatus type already expected data.device. 7 new tests in tests/server/test_health_controller.py (device field present/absent, reports cpu when loaded, _get_model_device controller/none paths). Regression-proven: removing the wiring fails test_health_has_device_field + test_health_reports_device_when_loaded. Live: /health model_loaded=true model_type=Qwen/Qwen2.5-0.5B-Instruct device=cpu. Full tests/server: 950 passed, 5 failed (pre-existing test_multimodal_router.py bugs: missing os import line 402, patch of non-existent domains.feedback.hf_dpo, analyze 500).
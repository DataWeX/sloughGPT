---
id: 20260808_112151_wire-processguard-into-production-model-loading
title: Wire ProcessGuard into production model loading
status: done
tags: process-guard,stability
created: 2026-08-08T11:21:51.460280+00:00
---

Wire ProcessGuard into production model loading

ProcessGuard wired into production model loading and verified. controllers/models.py builds/starts a ProcessGuard for HF models, lazy-guard autoload defers parent weight load when a .slnc is available, adopt_process_guard/get_process_guard_status/set_process_guard_enabled runtime toggles, _stop_process_guard cleanup. config.py runtime default enabled=true (SLO_ENABLE_PROCESS_GUARD). Verification: tests/server/test_startup_routers.py + test_models_controller.py 71 passed; packages/core-py test_process_isolation/hf_model_worker/kernel_process 26 passed.
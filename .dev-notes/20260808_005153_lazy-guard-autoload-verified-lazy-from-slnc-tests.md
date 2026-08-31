---
id: 20260808_005153_lazy-guard-autoload-verified-lazy-from-slnc-tests
title: Lazy-guard autoload verified + lazy_from_slnc tests
status: done
tags: inference,startup,lazy-guard
created: 2026-08-08T00:51:53.201806+00:00
---

Lazy-guard autoload verified + lazy_from_slnc tests

Verified the editor's lazy-guard autoload wave (startup.py refactor @01:43: _try_lazy_guard_autoload/_build_guard_for_model/_register_loaded/_shutdown_process_guard; config.py @01:38 SLO_LAZY_GUARD_AUTOLOAD/SLO_PROCESS_GUARD_MEMORY_LIMIT_MB; slonet_provider.lazy_from_slnc; controllers.adopt_process_guard/_stop_process_guard). No direct test existed for lazy_from_slnc (the feature's entry point) nor for the startup refactor. Added 5 tests to test_slonet_provider_real.py: header-only deferral (no weight pages, eager tokenizer, metadata total_params), lazy generate == eager generate (greedy), release_model->reload round-trip (deterministic same output), release-before-load (False), and cross-turn KV sessions via chat_stream session_id (active_sessions/cached_tokens/clear_session/clear_all_sessions). Result: test_slonet_provider_real.py 31 passed (26+5); test_process_guard_controller.py 11 passed; test_config.py 34 passed (prior). Editor moved on to a new web wave (useTrainingForm.ts, TrainingFormCard.tsx).
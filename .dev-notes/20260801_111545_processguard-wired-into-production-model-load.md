---
id: 20260801_111545_processguard-wired-into-production-model-load
title: ProcessGuard wired into production model load
status: done
tags: infra,processguard,slonet
created: 2026-08-01T11:15:45.175292+00:00
---

ProcessGuard wired into production model load

Added ProcessGuard delegation to SloNetServer (generate/generate_stream delegate to guarded subprocess when alive, in-process fallback otherwise; crash/restart callbacks wired to circuit breaker; guard health surfaced in metadata()). Added SloNetChatProvider.to_server(). setup_providers() accepts process_guard and auto-builds a guard-backed server. Wired both load paths: startup.py autoload passes process_guard to setup_providers; controllers/models.py _load_hf_model builds+starts a guard via ServerConfig.from_env (SLO_ENABLE_PROCESS_GUARD) and stops it on unload. Tests: 8 new SloNetServer delegation tests, 2 setup_providers wiring tests, 1 to_server test. All core-py suites pass.
---
id: 20260801_093108_wire-processguard-into-production-model-loading
title: Wire ProcessGuard into production model loading
status: done
tags: roadmap,process-guard,infra
created: 2026-08-01T09:31:08.478967+00:00
---

Wire ProcessGuard into production model loading

ProcessGuard wired into production SloNet model loading.

- startup.py _load_and_register passes process_guard to setup_providers; _load_hf_model builds/starts guard via _build_process_guard(model_id) (reads ServerConfig.from_env() -> enable_process_guard/quant kwargs; slnc_path = model dir / model.slnc; worker id slo-{model}; max_restarts=3, restart_delay=2.0, generate_timeout=120.0).
- SloNetServer supports own guard delegation (init param process_guard=None; generate/generate_stream delegate when _use_guard(); crash/restart callbacks wired to circuit breaker; metadata() includes process_guard health).
- SloNetChatProvider.to_server(process_guard=...) builds guard-backed server; setup_providers takes process_guard and _server_from_provider helper.
- Guard stopped on reload and in unload_model.
- 11 new tests (8 SloNetServer delegation + 2 setup_providers wiring + 1 to_server). Affected suites pass; py_compile clean.
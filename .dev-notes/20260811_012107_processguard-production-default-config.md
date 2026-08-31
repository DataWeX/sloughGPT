---
id: 20260811_012107_processguard-production-default-config
title: ProcessGuard production default config
status: done
tags: infra,process-guard,config
created: 2026-08-11T01:21:07.151237+00:00
---

ProcessGuard production default config

Closed roadmap #2. ProcessGuard wiring already existed on _load_hf_model (lazy+eager) and autoload paths; the gap was dead config: ServerConfig.enable_process_guard defaulted False / env 'false' but was never consulted (all paths read get_process_guard_enabled(), default 'true'). Made enable_process_guard the single source of truth: dataclass default True, env default 'true', and _runtime_process_guard_enabled now derives from ServerConfig.from_env().enable_process_guard (still overridable via POST /models/process-guard). Added tests/test_config_process_guard_default.py (6 tests). Verified: server tests 63 targeted pass, process-isolation suites 185 pass; full server suite failures unchanged vs baseline (pre-existing cross-test 429 rate-limit pollution). ROADMAP #2 struck through.
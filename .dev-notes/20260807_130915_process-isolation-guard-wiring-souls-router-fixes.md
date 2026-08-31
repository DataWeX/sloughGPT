---
id: 20260807_130915_process-isolation-guard-wiring-souls-router-fixes
title: Process isolation guard wiring + souls router fixes
status: done
tags: process-guard,models,autoload,startup,souls
created: 2026-08-07T13:09:15.185068+00:00
---

Process isolation guard wiring + souls router fixes

Session: process isolation guard wiring + souls router fixes.

DONE:
- startup.py: autoload gating on correct config; creates+adopts ProcessGuard; _shutdown_process_guard added.
- controllers/models.py: _build_process_guard reads ServerConfig (fixed NameError); _resolve_active_model_id; adopt/status/toggle.
- routers/souls.py: fixed HTTPException import + route order (/{soul_name} after static paths).
- NEW SloNetServer.set_process_guard(guard) swaps the guard reference + rewires circuit-breaker callbacks.
- NEW provider.py attach_process_guard_to_provider(guard) propagates a rebuilt guard to the slonet-native server.
- controllers/models.py: _build_process_guard and adopt_process_guard now call attach_process_guard_to_provider.
- Memory limit: default 4096MB reported over_limit=true permanently (worker RSS ~7.3GB). Added resolve_memory_limit_mb() in process_guard.py (explicit >0 wins, else auto max(8192, slnc_mb*8)); SLO_PROCESS_GUARD_MEMORY_LIMIT_MB env in ServerConfig; wired through all 3 creation sites (startup autoload, _build_process_guard, _build_process_guard_for_path). Live: limit=19231MB, over_limit=false.
- Tests: 11 in test_process_guard_controller.py (incl. runtime-rebuild regression + memory-limit + env parse); 4 resolve_memory_limit tests in core-py; full server suite 311 passed.

ROOT CAUSE FOUND (this session): after runtime disable→re-enable, generation still ran in-process (requests_served stayed 0) because the slonet-native SloNetServer kept referencing the old stopped guard. Fixed by propagating the rebuilt guard to the server.

VERIFIED live on port 8000:
- autoload: guard active, text=slonet-native, non-streaming AND streaming (/chat/stream SSE tokens) both route through the guard subprocess (requests_served increments).
- disable: active=False. re-enable: new worker, requests_served=0. generate: requests_served=1 (delegation restored).
- guard wiring isolated run: requests_served=3 (autoload path), then rebuild scenario requests_served=1 (REBUILD_WIRED).
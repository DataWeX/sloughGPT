---
id: 20260807_170402_lazy-guard-backed-model-loading-slonet
title: Lazy guard-backed model loading (SloNet)
status: done
tags: inference,process-guard,lazy-load
created: 2026-08-07T17:04:02.656447+00:00
---

Lazy guard-backed model loading (SloNet)

    --- 2026-08-08 (cont. 2) StatusBar fix + useLiveStatus fallback dedup + file-loss recovery ---
    
    STATUSBAR FOOTER BUG (fixed earlier this session):
    - Root cause: apps/web/components/StatusBar.tsx soul-fetch effect deps [health, connectionStatus]; SSE /health/stream pushes a fresh health object every 3s -> soulsController.getCurrent()+getTraitWeights() fired 2 un-cancelled HTTP requests every 3s forever. Under load, hung requests pile up.
    - Fix: gate effect on stable values (connectionStatus string + health.soul string) + cancelled cleanup flag. Runs once per connection/soul; re-fetches only on soul switch/reconnect. StatusBar.test.tsx +2 (no re-fetch on health-only ticks; re-fetch on soul change). 11/11 pass.
    
    useLiveStatus fallback-poll dedup (this turn):
    - useLiveStatus.ts: removed immediate startFallbackPoll() on mount (duplicate /health/detailed request alongside SSE on every app load). Fallback now starts only after a FALLBACK_POLL_MS (8s) grace window if no SSE health event arrived, or immediately on onClose. Added fallbackDelayTimer + _receivedHealthEvent flag; stopFallbackPoll clears both. poll() null-path now calls checkReload() (was only in catch path). onClose clears pending delay timer before starting poll.
    - Tests +2 (fake timers): no poll within grace window when SSE delivers; poll starts after 8s without SSE event. 19/19 pass.
    
    INCIDENT — file truncation + recovery:
    - The onClose edit tool call was interrupted mid-write; apps/web/hooks/useLiveStatus.ts became 0 bytes. git checkout restored HEAD (288 lines) but the working tree had ~69 uncommitted lines (mapDetailedToSnapshot + initLiveStatus fallback from the 2026-08-06 monitoring expansion) which were LOST.
    - Recovered by reconstructing the full file from the session's earlier Read output (exact 357-line working-tree content) + re-applying the dedup edits. Verified: tsc exit 0; useLiveStatus.test.ts 19/19 (imports + tests mapDetailedToSnapshot); StatusBar 11/11; AppLayout 13/13.
    - LESSON: git checkout -- <file> on an uncommitted file destroys working-tree-only changes. Restore from editor/tool output or vitest transform cache first; use git stash/diff before clobbering.
    
    Stack: API died silently (parent gone, guard worker orphaned) — killed orphans, rebooted. api 200 model_loaded Qwen, web 200, SSE ticks 2-3 events/8s.
  
  --- 2026-08-08 (cont. 3) Hang-drill verification + GOLD + EBADF race + config cleanup ---
  
  Drill instruments fully verified live against rebooted stack:
  - faulthandler stack dumps firing every 20s in ~/api_drill.log (SLO_DUMP_STACKS=1)
  - [SLOW] middleware: 21 lines under benchmark load, all ~1.6s 200 OK — no 5xx, no hang signatures
  - web-vitals chain verified end-to-end: WebVitals.tsx (warning-level) -> lib/dev-log.ts batch (5s/20) -> POST /errors/logs/ingest {logs:[...]} -> OutputBuffer -> /system/output. Confirmed record 'slo.web.web-vitals web-vitals LCP' appears. (My earlier probe used wrong shape; frontend contract is {logs:[...]}.)
  - benchmark_stability.py --runs 20 -> GOLD STANDARD 100/100 (0 crashes 0/20, latency degr 1.06x, empty 0%, CV 0.00, response 100%). Qwen2.5-0.5B on CPU avg 1616ms/req.
  - /errors/recent clean (0 errors) on fresh boot.
  
  Transient startup race — [Errno 9] Bad file descriptor:
  - One boot (01:41) failed _phase6_routers with EBADF; orchestrator logged 'Lifecycle startup incomplete - continuing anyway' and served with ONLY pre-registered health/status routes (5 routes; /models /souls /system/* all 404). Reboot succeeded cleanly (44 routes). Intermittent (1/2), non-deterministic.
  - Fixes/defense: startup.py _phase6_routers except now logs exc_info=True so next occurrence captures the traceback. Root cause still unconfirmed (suspect fd/pipe race between guard subprocess spawn + stdio bridge + imports); watch next boot.
  
  config.py _SKIP_ENV_KEYS:
  - Added SLO_DUMP_STACKS + SLO_DUMP_STACKS_INTERVAL (drill-instrument env consumed by main.py faulthandler timer). Boot now clean: 0 'Unknown config key' warnings (verified via /system/output).
  
  Checks: py_compile OK (startup.py, config.py); tsc --noEmit exit 0; vitest StatusBar 11/11, AppLayout 13/13, useLiveStatus 19/19.

  --- 2026-08-08 (cont. 4) EBADF startup race root-caused + fixed ---
  
  Full root cause captured via faulthandler all-threads dump (boot_7.log, thread stacks at failure moment):
  - TWO threads doing first-time module imports concurrently: background model-load thread (_load_and_register -> _try_lazy_guard_autoload at startup.py:621 importing domains.infrastructure.process_guard / domains.api.sse_envelope) + main thread (_phase6_routers -> get_all_routers importing routers). Both threads hit importlib get_data -> OSError [Errno 9] Bad file descriptor on 'packages/core-py/domains/api' (a NAMESPACE package — no __init__.py). Also observed: 'cannot import name ProcessGuard from domains.infrastructure.process_guard' (partial import race). Consequence when the race wins: orchestrator continues ('Lifecycle startup incomplete'), API serves only pre-registered health/status routes (/models /souls /system/* -> 404).
  - domains/api reached lazily: routers import domains.api.sse_envelope at module level; training_queue imports it inside function bodies (lazy) — my earlier prewarm missed it.
  
  FIX (two layers, startup.py):
  1. Prewarm the background thread's full import graph in the main thread BEFORE the model-load task is created (_preload_model_imports() + _PREWARM_MODEL_LOAD_IMPORTS: state, config, safetensors_loader, slonet_provider, process_guard, server_state, controllers.models, model_registry, models.provider, slo_manager, slolib.gpu, model_catalog, task_queue, training_queue, domains.api.sse_envelope). Thread's later imports become cached sys.modules hits — zero fresh file reads during router registration.
  2. _phase6_routers retries once on transient ImportError / OSError errno 9 (importlib rolls failed imports out of sys.modules, so a retry re-imports cleanly; already-included routers are skipped). All-threads faulthandler dump + print_exc now emitted on the FIRST attempt too (forensics before the retry).
  
  VALIDATION: 54 total boots. With fix: 10/10, then 20 with ONE transient EBADF that the retry RECOVERED to 44 routes + 'Startup complete' (previously that failure degraded the API to 5 routes), then 24/24 clean. Restored drill boot: 44 routes, /health /models /souls /system/output /errors/recent all 200. benchmark_stability.py --runs 20 -> GOLD STANDARD 100/100 (unchanged). Server tests: 80/80 pass.
  
  Boot-loop tooling: /tmp/opencode/bootloop.sh (rebuildable — /tmp is cleared) probes /models AFTER 'Startup complete' in the log; /models -> 200 discriminator (422 during the retry window was a probe-timing artifact, not a real state).
  
  Checks: py_compile OK (startup.py); 80/80 server router tests; tsc/vitest untouched (no frontend change).
---
id: 20260813_153323_fix-log-formatting-and-coloring-in-main-binary
title: Fix log formatting and coloring in main binary
status: done
tags: logging,bridge,console-logger,colors
created: 2026-08-13T15:33:23.373363+00:00
---

Fix log formatting and coloring in main binary

Telemetry logs were not formatted or colored in the main binary.

ROOT CAUSE 1 (formatting): BridgeHandler.emit() (bridge.py:77-79) only reads extra['context'] (dict). The generation-telemetry logs passed structured fields (mode, elapsed_ms, prompt, result) as top-level extra keys, which stdlib copies onto the LogRecord but the bridge never reads -> dropped. Fixed: nested all structured fields under extra['context'] in slonet_server.py (x2), model_worker.py, inference.py.

ROOT CAUSE 2 (colors): _Ansi constants were computed at import time as '' when stderr was not a TTY (piped/redirected), so even ConsoleLogger(colors=True) emitted no ANSI. Auto-detect was also frozen at import. Fixed in console_logger.py: _Ansi codes are now unconditional; _default_color_enabled() evaluates NO_COLOR / SLO_LOG_COLOR / FORCE_COLOR / stream.isatty() at construction time.

Also wired ModelMetrics.last_request_time into record_success()/record_failure() (was in snapshot but never set) and updated test_slonet_server.py snapshot contract + added 2 regression tests (colors=True without tty; structured fields must live under context).

Verification: logging + bridge + slonet_server + server integration suites pass; py_compile clean; tsc exit 0.
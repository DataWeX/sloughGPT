---
id: 20260808_114000_thicken-testsserver-suites-errors-router-40
title: Thicken tests/server suites: errors router +40
status: done
tags: testing,server,router
created: 2026-08-08T11:40:00.875665+00:00
---

Thicken tests/server suites: errors router +40

Errors router 21->40 passed. Added: /errors/logs/ingest (single/multi/level-mapping/unknown-level-defaults-info/exception-context/empty batch; patch at domains.infrastructure.output_buffer.get_server_buffer), GET /errors/log (opencode log entries), validation 422s (message>5000, 101 errors, missing errors/logs fields), method 405s (recent/grouped/trends/export/unread/log/ingest/clear). Full tests/server: 1379 passed, 68 deselected (~69s).
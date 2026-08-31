---
id: 20260730_160925_logbuffer-bugfix-tests
title: LogBuffer bugfix + tests
status: done
tags: shell,logging,bugfix
created: 2026-07-30T16:09:25.782183+00:00
---

LogBuffer bugfix + tests

Fixed LogBuffer __len__ falsy bug (buffer or get_log_buffer -> identity check). Wrote 17 tests for log_buffer.py. Added handler to child loggers (slo.kernel, slo.shell.runtime, slo.shell.init) that disable propagation during boot. Added --stats (level distribution, top sources, time range) and --export FILE flags to _cmd_logs. Wrote 10 new shell tests for _cmd_logs covering empty, clear, display, level/source filter, limit, stats, export. All 350 shell+wm+log tests pass.
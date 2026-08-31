---
id: 20260813_160841_structured-logging-auto-capture-extra-fields-in-bridge
title: Structured logging: auto-capture extra fields in bridge
status: done
tags: logging,bridge,telemetry
created: 2026-08-13T16:08:41.538323+00:00
---

Structured logging: auto-capture extra fields in bridge

BridgeHandler + BufferLogHandler silently dropped non-standard stdlib extra fields (e.g. extra={'mode','elapsed_ms','total_images'}) because they only exist as LogRecord attributes. Added record_extra_context() in domains/logging/bridge.py which merges explicit context + auto-captures stray extra fields (mirrors Logger **ctx kwargs). BufferLogHandler (SSE path) reuses the helper. JSONFormatter already handled record.__dict__; shell LogBufferHandler is a plain CLI view (out of scope). All 79 stray sites across core-py (5) + apps/api/server (74) now render as key=value. Added regression tests: test_logging_bridge (auto-capture + explicit-wins), test_output_buffer TestBufferLogHandler (stray capture, explicit-wins, mixed tag/context/stray production shape). 226 affected tests pass; ruff clean on changed files (pre-existing E741 at output_buffer.py:167 untouched).
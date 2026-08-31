---
id: 20260803_085219_wave-s-event-bus-to-100-2-real-bugfixes
title: Wave S: event_bus to 100% + 2 real bugfixes
status: done
tags: infra,coverage,event_bus
created: 2026-08-03T08:52:19.203940+00:00
---

Wave S: event_bus to 100% + 2 real bugfixes

event_bus.py 156 stmts 0 miss (100%), was 88% (18 misses). Real bugfix 1: once('*') wildcard subs were never auto-removed - emit/emit_sync called self.off(event, handler) with the concrete event name so off() looked in _subscriptions[event] and left the wildcard in place, firing on every subsequent emit. Fixed by tracking each sub's target as (target, sub) pairs and calling off(target, handler). Real bugfix 2: install_log_subscriber _log_handler used 'EVENT %s%s%s' with only 2 format args - always raised TypeError on first event; fixed to 'EVENT %s%s'. Added 9 tests (TestEventBusCoverage): once wildcard fires once (n1=1, n2=0), once specific sync, emit_sync errored once handler logs + removes, replay async handler warns (filterwarnings on never-awaited coroutine), replay errored handler logs, _is_noisy True/False, install_log_subscriber (captures EVENT, skips heartbeat, idempotent guard, default-bus via get_event_bus). Also added filterwarnings to pre-existing test_emit_sync_skips_async_handlers to make run warning-free. Sweep: 490 passed (was 481), only pre-existing test_lifecycle_endpoint fastapi failure. pycache cleared, py_compile OK. Board was 141 cards.
---
id: 20260804_081442_kv-session-lru-cap-monitoring-surfacing
title: KV session LRU cap + monitoring surfacing
status: done
tags: inference,kv-cache,lru
created: 2026-08-04T08:14:42.247081+00:00
---

KV session LRU cap + monitoring surfacing

Completed: added _kv_max_sessions (default 64, kv_max_sessions param) + _evict_lru_session() to SloNetChatProvider; _resolve_session_kv evicts LRU other-session over cap; session_stats exposes max_sessions; health kv_sessions + monitoring card surface LRU cap. 6 new tests (30 in test_slonet_session_ttl). 161 core + 44 app tests pass; tsc clean; system-controller 13 tests pass; pycache cleared; plan doc updated.
---
id: 20260804_075839_cross-turn-kv-server-session-id-threading-thread-safe-sessio
title: Cross-turn KV: server session_id threading + thread-safe session map
status: done
tags: inference,kv-cache,thread-safety
created: 2026-08-04T07:58:39.448395+00:00
---

Cross-turn KV: server session_id threading + thread-safe session map

Completed: server layer now threads session_id into KV reuse on both streaming and non-streaming paths; ChatDomain respond->_generate forwards session_id; stack benchmark (TestStackCrossTurn) verifies cached tokens 0->54 across 3 turns; session KV map made thread-safe with threading.Lock guarded resolve/evict/stats/clear; 3 concurrency tests added (24 in test_slonet_session_ttl); 155 core + 44 app tests pass.
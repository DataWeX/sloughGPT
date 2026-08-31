---
id: 20260808_112845_router-suite-blitz-9-companionhealthlearner-kb-stats-hang-fi
title: Router suite blitz 9 — companion/health/learner + kb_stats hang fix
status: done
tags: tests,server,routers
created: 2026-08-08T11:28:45.831915+00:00
---

Router suite blitz 9 — companion/health/learner + kb_stats hang fix

Router/controller blitz batch 9. Companion 20→40: full-replacement semantics, partial patch (nulls ignored), trait bounds 422s, name/preset length 422s, unknown preset OK, message>10000 422, no-provider error response, wrong-method 405s. Health 20→29: /health/stream SSE tests (patched fastapi.Request.is_disconnected AsyncMock [False,True] + asyncio.sleep AsyncMock to terminate; build-exception→no data), method-mismatch 405s. Learner 20→39: /learn/ingest-url, conversations as bare-array body (FastAPI binds list[list[str]] to body, wrapped-key body→422), short-pair skip, feed subscribe/unsubscribe statuses, validation 422s (missing query, poll_interval<60, top_k>100), method 405s. BUGFIX kb.py knowledge_stats: old batching loop incremented offset but never passed it to list_all (no offset param) → infinite loop when store >200 facts (hung full suite at test_server_api). Rewrote to single pass list_all(top_k=5000); added regression test test_stats_with_many_facts_terminates (250 facts, asserts totals/counts). Full tests/server: 1348 passed, 68 deselected (~101s). pycache cleared.
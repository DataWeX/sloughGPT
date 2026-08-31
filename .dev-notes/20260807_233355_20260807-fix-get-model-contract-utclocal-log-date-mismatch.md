---
id: 20260807_233355_20260807-fix-get-model-contract-utclocal-log-date-mismatch
title: 20260807 Fix _get_model contract + UTC/local log-date mismatch
status: done
tags: 
created: 2026-08-07T23:33:55.539495+00:00
---

20260807 Fix _get_model contract + UTC/local log-date mismatch

Editor's slonet_provider.py lazy-_get_model factory broke stub providers that bind SloNetChatProvider._resolve_session_kv: added _get_model() to _StubProvider (test_slonet_kv_benchmark.py, 8 passed) and _StackProvider (scripts/benchmark_kv_reuse.py, 44 passed). Third binder test_slonet_session_ttl already had it. Also fixed pre-existing UTC/local date mismatch in domains/chat/domain.py: _log wrote responses_YYYYMMDD.jsonl with UTC date, get_recent_responses/get_stats read with local date - diverges at UTC-midnight. Reader now uses UTC. test_chat_domain.py: 21 passed. Fresh 284-file sweep running in background.
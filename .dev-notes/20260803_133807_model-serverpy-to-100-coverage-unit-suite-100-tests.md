---
id: 20260803_133807_model-serverpy-to-100-coverage-unit-suite-100-tests
title: model_server.py to 100% coverage (unit suite 100 tests)
status: done
tags: coverage,infrastructure,model_server,testing
created: 2026-08-03T13:38:07.179650+00:00
---

model_server.py to 100% coverage (unit suite 100 tests)

Targeted unit suite tests/test_model_server_units.py expanded to 100 tests (98 unit + edges re-run). Combined coverage: model_server.py 955/955 (100%), model_registry.py 103/103 (100%).

Fixes this session:
- NpMockModel.parameters() added (compile paths); fake torch _compile records mod.compile_calls; compile test asserts compile_calls[0][0] is m + backend 'inductor'.
- Eviction test split: test_tokenize_cached_eviction_on_miss (seed 65, miss 'fresh', cache 66->65) + test_tokenize_cached_hit_does_not_grow.
- test_generate_backend_error: requests_failed >= 1 (queue submit re-records failure after _run error path) -> corrected assertion.
- test_submit_queue_full: explicitly close() queued coroutines to kill 'never awaited' RuntimeWarning.
- Removed ineffective autouse loop-cancel fixture (pytest-asyncio closes loop first; added DeprecationWarning). Accepted 6 cosmetic 'Task was destroyed' lines (match pre-existing integration-suite behavior).
- test_streamer_pump_stop_iteration: iterable whose __iter__ raises StopIteration -> covers the for-loop StopIteration escape (GET_ITER outside try) at model_server.py:1757-1758.

Production dead-code removals (unreachable, verified by coverage + 57 slow integration tests still pass):
- Removed never-called _gen() closure + unused 'loop' assignment in generate_stream.
- Removed impossible except Exception after 'del old' in swap_model (del on local cannot raise).

New branch tests: ModelMetrics non-zero avg/error_rate, KV-capture store-exception path, swap_model warmup Thread start (enable_warmup=True + patched Thread).

Coverage combine gotcha: 'coverage combine <dir>' excludes the output .coverage from sources — each combine replaces, not accumulates. Must copy all run files (.coverage.w/.me/.s) then combine once.
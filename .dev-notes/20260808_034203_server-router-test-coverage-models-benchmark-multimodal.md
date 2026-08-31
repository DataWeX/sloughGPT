---
id: 20260808_034203_server-router-test-coverage-models-benchmark-multimodal
title: Server router test coverage: models, benchmark, multimodal
status: done
tags: server,tests,coverage,routers
created: 2026-08-08T03:42:03.483064+00:00
---

Server router test coverage: models, benchmark, multimodal

Server router test coverage: models, benchmark, multimodal.

MODELS (done, prior session): fixed int-from-float 422 in list_hf_models + list_models coercion; 3 regression tests; 2 load-event tests.

BENCHMARK (this session):
- Deduplicated test_benchmark_router.py: removed stub subset classes (first TestRunBenchmark/TestGetMetrics/TestQuality/TestLoggedResponses/TestTrackerStats, lines 25-98) that duplicated the expanded set. 32 -> 22 tests, all 7 endpoints covered (run, metrics, perplexity, quality, responses, stats, history/clear).

MULTIMODAL (this session):
- Added /multimodal/train-video coverage: 409 when already running, success (job started, executor.submit called), 422 on missing data_path.
- Added /multimodal/video-infer success path (generates caption from latest checkpoint, forwards max_len/temperature, asserts load_checkpoint called).
- Added /multimodal/pdf/upload coverage: text_extract method, vlm method, per_page pagination join.
- Added /multimodal/process-video coverage: success (caption, num_frames, engine.vision/generate), 500 when engine missing.
- Full 17-endpoint route map now covered.

RESULTS:
- test_multimodal_router.py + test_benchmark_router.py: 63 passed.
- tests/server full: 1048 passed, 2 failed (pre-existing, untracked file test_startup_routers.py event-loop teardown on Python 3.12.3, unrelated), 68 deselected.
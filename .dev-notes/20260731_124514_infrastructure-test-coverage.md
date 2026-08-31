---
id: 20260731_124514_infrastructure-test-coverage
title: Infrastructure test coverage
status: done
tags: infrastructure,tests
created: 2026-07-31T12:45:14.092126+00:00
---

Infrastructure test coverage

Added test coverage for previously untested infrastructure modules in packages/core-py. New suites: test_truth_labeler (27), test_truth_maintainer (19), test_anchor_store (33), test_watchdog (16), test_mps_monitor (25), test_model_protector (25), test_spaced_repetition (21), test_point_compressor (10). All pass; batch of 10 files (incl. pre-existing test_tags, test_meaning_points, test_config) green. Source fixes: truth_labeler descriptive score 0.6->0.3 and X-is-Y now requires article; SloAdam only updates requires_grad params (test-side fix). Known blockers unchanged: psutil/starlette missing deps, HF gpt2/Qwen2-0.5B models not cached (test_numpy_engine, 3 test_point_library cases).
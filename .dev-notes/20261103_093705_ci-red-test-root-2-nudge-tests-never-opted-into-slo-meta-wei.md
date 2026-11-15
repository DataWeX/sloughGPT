---
id: 20261103_093705_ci-red-test-root-2-nudge-tests-never-opted-into-slo-meta-wei
status: done
board: 20261103_093705_ci-red-test-root-2-nudge-tests-never-opted-into-slo-meta-wei
title: "CI red test-root: 2 nudge tests never opted into SLO_META_WEIGHT_NUDGE (93f3b1196 ships sampler nudge dark)"
tags: ["ci", "tests"]
sprint: ""
gh: null
---

# CI red test-root: 2 nudge tests never opted into SLO_META_WEIGHT_NUDGE (93f3b1196 ships sampler nudge dark)

Run 8d7191279 (2026-11-03): lint success, test-core-py success (both prior fixes verified), test-root FAILURE -> build SKIPPED (build needs test-root, so the web Dockerfile still has no CI verdict).

FAILING (reproduced locally, env -u PYTHONPATH):
  tests/test_inference_generate.py::TestGenerateEndpoint::test_generate_nudges_params_left_at_default  - assert 0.7 == 0.77
  tests/test_inference_generate.py::TestGenerateEndpoint::test_generate_telemetry_records_sent_temperature - assert 0.7 == 1.05

ROOT CAUSE: 93f3b1196 (card 20261103_071231, done) gates _apply_meta_weights behind SLO_META_WEIGHT_NUDGE (default dark = 0) with the comment 'feedback must steer context, not sampling'. Its NEW suites opt in via fixture (tests/test_meta_weights_clamp.py autouse _nudge_enabled sets SLO_META_WEIGHT_NUDGE=1), but the two PRE-EXISTING endpoint tests in tests/test_inference_generate.py were never updated -- they patch domain.feedback.get_meta_weight_manager and assert the nudge applies unconditionally. First red at e3fce3a01 (the run right after the landing); test-root was green at e0ac6ab16 before it.

FIX (test-only): give the two tests an opt-in monkeypatch.setenv('SLO_META_WEIGHT_NUDGE','1') mirroring the landing's own fixture, plus a docstring note that the dark default is covered by TestNudgeIsDarkByDefault. No production code change.

CLOSED 2026-11-03: fixed in 5fa180c03 — both tests opt into SLO_META_WEIGHT_NUDGE=1 (test-only); test-root green on the real runner.

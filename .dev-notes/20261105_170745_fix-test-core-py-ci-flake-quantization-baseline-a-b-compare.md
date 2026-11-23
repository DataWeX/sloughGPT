---
id: 20261105_170745_fix-test-core-py-ci-flake-quantization-baseline-a-b-compare
status: done
board: 20261105_170745_fix-test-core-py-ci-flake-quantization-baseline-a-b-compare
title: "Fix test-core-py CI flake: quantization-baseline A-B compare trips on runner perf noise (test_baseline_cli_json_block)"
tags: ["ci", "flaky", "tests", "benchmarks", "quantization"]
sprint: ""
gh: null
---

# Fix test-core-py CI flake: quantization-baseline A-B compare trips on runner perf noise (test_baseline_cli_json_block)

chore(ci): stabilize test_baseline_cli_json_block — same-machine A-B baseline compare trips on runner perf noise

EVIDENCE (measured 2026-11-05):
- CI run 42061785501 (e601cf6d8): `tests/core-py/test_quantization_benchmark_e2e.py::TestBaseline::test_baseline_cli_json_block` FAILED — `assert data["baseline"]["regressions"] == []` got `[{'key': 'tiny…', 'current': 5}]`. Suite otherwise: 1 failed / 42801 passed / 21m11s.
- Same test PASSED in CI at a9964c422 (run earlier the same day, 13:40) and the test file is unchanged since 815cb94c9 — so this is not a code change.
- Local repro: 1 passed / 57s. Does not reproduce outside a loaded CI runner.
- The change at that SHA was in-process (tests/conftest.py teardown); the failing test drives the CLI as a subprocess — no plausible causal path.

ROOT CAUSE (diagnosis complete):
- The test does a same-machine A-B: run 1 writes a fresh baseline into tmp_path, run 2 (seconds later) compares against it — see tests/core-py/test_quantization_benchmark_e2e.py:1041-1052 (also test_baseline_cli_creates_then_validates, same pattern, same exposure).
- `regressions != []` between two back-to-back quick runs means the SECOND run measured slower than the first — pure runner performance noise (noisy shared CI CPU, thermal, co-tenancy), not a real regression.
- Compare logic: scripts/benchmark_quantization.py:1513 `_compare_baseline` — deltas are flagged as regressions at :1540; threshold is effectively zero-tolerance for these metrics.

FIX DIRECTIONS (pick one at implementation time):
1. Tolerance: allow a small relative delta (e.g. ±10-15%) before flagging a regression in the e2e path — real regressions of interest are far larger than runner noise.
2. Test-side: assert on the baseline block shape (`exists` flag, `regressions` key present) rather than requiring zero regressions in an A-B that cannot distinguish noise from drift.
3. If keeping strict compare: retry the compare once on failure (flake retry) before failing.

IMPACT: intermittent red on main CI for test-core-py; each occurrence costs a triage cycle (this one was fully diagnosed). Not blocking — flaky signal, not a real quantization regression.

CHECK IT LANDED: two consecutive CI test-core-py runs on main pass this test; no further triage notes for this test id in AGENT_SYNC.

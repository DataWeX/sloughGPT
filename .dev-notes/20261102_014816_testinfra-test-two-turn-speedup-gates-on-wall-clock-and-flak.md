---
id: 20261102_014816_testinfra-test-two-turn-speedup-gates-on-wall-clock-and-flak
status: done
board: 20261102_014816_testinfra-test-two-turn-speedup-gates-on-wall-clock-and-flak
title: "test(infra): test_two_turn_speedup gates on wall clock and flaked at 0.62x — assert recomputed tokens instead"
tags: ["inference", "tests", "benchmark", "flake"]
sprint: ""
gh: null
---

# test(infra): test_two_turn_speedup gates on wall clock and flaked at 0.62x — assert recomputed tokens instead

FOUND 2026-11-02 while landing card 20261102_012822 (identical-prompt KV resume).

test_slonet_kv_benchmark.py::TestCrossTurnBenchmark::test_two_turn_speedup ends with:
    assert total_cached <= total_fresh * 1.5, "KV cache should not be slower"
i.e. it gates on WALL CLOCK of a 1-block, 64-dim model where a single turn is a few milliseconds. Observed once in a combined run: 0.62x -> FAILED. Then: 3/3 passes alone, 2/2 passes in the exact same combined set, and a pass on the pre-change tree (origin/main) running the same four files. So it is scheduling noise, not a regression — but it fails for reasons unrelated to the property it names, which is the same "gate that does not test what its name says" trap as the ported reset_accelerator test.

WHY IT IS THE WRONG GATE FOR A CACHE: a cache's correctness property is HOW MUCH WORK IT SKIPPED, which is exact and cheap to assert; elapsed time is an environment measurement with a noise floor comparable to the signal on a tiny model.

PROPOSED: assert on recomputed tokens instead of milliseconds. _resolve_kv_state already returns `start` and `_prefill = prompt_len - start`; expose the per-call recomputed length (a counter on the state, or read kv_len before/after) and assert:
    turn2_recomputed == 1        (extension by exactly 1 token)
    repeat_recomputed == 1       (full-prompt match, after 20261102_012822)
    cold_recomputed == prompt_len
Keep printing the timings for humans (the suite's docstring already says it is a benchmark, run with -s), just do not assert on them. No threshold, no flake.

SCOPE: tests/core-py/test_slonet_kv_benchmark.py only. Do not touch generate/generate_numpy — the counter must be observation, not new production state, if it can be avoided (kv_len and the returned start already carry the answer).

CLOSED 2026-11-03: fixed by e0ac6ab16 — asserts recomputed token counts instead of wall clock; 60 passed x3, ruff clean.

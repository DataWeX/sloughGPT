---
title: "slonet benchmark: three results could not fail, one had never run"
date: 2026-10-11
status: done
branch: fix/slonet-benchmark-gates
landed: dadfad34e
tags: [benchmark, slonet, accelerator, cuda, gates, regression]
---

# Outcome

`scripts/benchmark_slonet.py` printed `Passed: X/Y` and `🏆 All tests
passed!` and wrote its results to `data/eval_results/slonet_benchmark.json`,
so every `TestResult` read as a **gate** — something that could go red and
therefore tell you a regression happened. Four things were wrong with that,
all in card `a25e490b`'s family.

Landed as **`dadfad34e`** on `fix/slonet-benchmark-gates`.

| Defect | What it meant |
|---|---|
| `accelerator_speed` appended `passed=True` literally | PASS at any ratio, including an accelerator *slower* than numpy |
| `accelerator_matmul_crossover` appended `passed=True` literally | same |
| `lstm_forward_backward_speed` appended `passed=True` literally | a raw timing with no baseline, so no invariant existed to check |
| `test_accelerator_matmul_crossover` never called from `run_all()` | the dispatch-gate check had **never once executed** |

The card's own 2026-10-10 note had caught the first one — *"passes at any
ratio including <1x — it can never fail; treat as a report, not a regression
guard."* The AST sweep found all three, plus the dead method, which nobody
had noticed because it still looked like coverage in the file listing.

# What was done

- **`accelerator_speed`** now gates on the invariant it already measured:
  `speedup > 1.0`. Marked `informational` when `acc.name == "cpu"`, because
  there the "accelerator" *is* numpy and speedup sits at ~1.0 by construction.
- **`accelerator_matmul_crossover`** gates on the condition its own
  diagnostic already spelled out in prose — `crossover <= dispatch_gate` —
  with `crossover=None` (accelerator never won at any size) treated as the
  worst case rather than a pass. It already early-returns on the cpu backend,
  so a real accelerator is guaranteed by the time it computes.
- **`lstm_forward_backward_speed`** is declared `informational=True`: kept in
  the results (the history is worth having) but excluded from the gate count.
  Inventing a threshold would have replaced a blind gate with a flaky one.
- **Wired into `run_all()`**, so the check actually runs.
- `print_summary()` counts **gates**, not results; `main()` takes its JSON
  summary from the same `gate_counts()` helper so the two cannot drift —
  previously the file said 8/8 while the terminal said 7/7 gates.
- Measurements print `○ MEASURE` instead of `✓ PASS`.

# Measured (GTX 1050 + cuPy 14.2)

```
accelerator_speed              17.4x  cuda vs numpy on (128,128)   ✓ PASS
accelerator_matmul_crossover   16,384 elements, gate=16,384        ✓ PASS
                              "Gate sits above the crossover: well placed."
lstm_forward_backward_speed    0.0974 s/step                      ○ MEASURE
```

The gate at 16,384 sits exactly on the measured crossover — the dispatch
threshold is well placed, and is now *checked* rather than asserted.

# Guards added

`tests/contract/test_benchmark_gates.py`, written test-first (red first, 5
failing on exactly the three named offenders):

1. no literal `passed=True` outside a declared measurement;
2. every result either computes its verdict or declares itself a measurement;
3. the summary counts gates and labels measurements;
4. the trophy is over gates;
5. the `informational` flag survives `asdict()` into the saved JSON;
6. **no benchmark method may be unreachable from `run_all()`** — this is the
   generalisable guard; it would have caught the dead Test 4b immediately;
7. `gate_counts()` excludes measurements;
8. the JSON summary is not recomputed from every result.

# Gates

`8 passed` (new) · `17 passed` (whole `tests/contract/`) · benchmark
`exit 0` at `7/7 gates + 1 measurement` with the JSON summary agreeing ·
`ruff check` and `ruff format --check` clean · mypy `1 -> 1` error, none new.

# Note for whoever reads this next

Two of these are the *same* defect a second time: a number wearing a
verdict's clothing. If a new benchmark result cannot name the invariant it
would fail on, it is a measurement — say so with `informational=True`, and
let `test_run_all_calls_every_benchmark_method` and the AST guard catch the
ones that drift.

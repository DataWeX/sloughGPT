---
title: latency records filed as "served" — model fetched from /health then dropped
date: 2026-10-09
status: done
branch: feat/aml-stream-loader
tags: [benchmark, latency, attribution, provenance]
---

# Symptom

`history --kind latency` showed two runs with **different model names** for
what was the same server:

```
2026-10-09T18:05:48  served                    mean=7919.1
2026-10-09T09:21:41  Qwen/Qwen2.5-0.5B-Instruct  mean=3102.6
```

and `compare` scored one against the other, reporting the difference as a
regression. Comparing a model against a placeholder is not a measurement of
anything.

# Root cause

`benchmark_latency.py` **already fetched the model** — `_host_context` path,
`urllib` call to `/health`, which returns `data.model_type` =
`"Qwen/Qwen2.5-0.5B-Instruct"`. It was placed in the result dict, written into
the baseline file, and **never printed**.

`benchmark_results._run_latency` does not read that dict. It shells out and
**scrapes stdout**. So the model never crossed the boundary, and `do_record`
fell through to `args.model or "served"` for every single run.

One value, fetched correctly, dropped on the floor between two files — the same
printer/parser seam as the `record --kind latency` parse bug.

# Fix

- `benchmark_latency.py` prints `  model: <name>` alongside mean/min/p50/p95.
- `_run_latency` captures it — **before** the float branch, deliberately:
  `_first_float` is greedy, so `"Qwen/Qwen2.5-0.5B-Instruct"` yields `2.5`,
  which is a perfectly ordinary-looking latency in milliseconds. Letting a
  model name into the float branch would have recorded a name as a
  measurement. `model: unknown` (the printer's fallback) is discarded rather
  than recorded.
- `do_record` prefers `args.model` → scraped model → `"served"`.

# Note on the constant

`_LATENCY_STDOUT` in the test file is copied **verbatim** from the printer and
had to be re-copied twice today (headroom line, now the model line). That is
the point of copying it rather than building it in the test: each printer
change forces the constant to be reconciled, which is what the previous drift
bug was missing. Comment in the test file says so explicitly.

# Files

- `scripts/benchmark_latency.py` — print `  model:`
- `scripts/benchmark_results.py` — `_run_latency` string branch, `do_record` fallback chain
- `packages/core-py/tests/test_benchmark_results.py` — re-copied constant, 2 new tests

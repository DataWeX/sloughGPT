---
title: benchmark_results `record --kind latency` parsed nothing — unit suffix vs bare float
date: 2026-10-09
status: done
branch: feat/aml-stream-loader
tags: [benchmark, latency, parsing, regression, tests]
---

# Symptom

`scripts/benchmark_results.py record --kind latency` printed a full measurement
**and** `[OK] Within acceptable range (±20%)`, then exited **1**, and
`history --kind latency` still showed one run. Success message, failure exit
code, nothing persisted — three facts that disagree, which is how it went
unnoticed: the visible output looked healthy.

# Root cause

`_run_latency()` shells out to `scripts/benchmark_latency.py` and parses its
stdout. Two files, two assumptions:

- **printer** (`benchmark_latency.py:161`): `mean: 2395.8 ms  (15 samples)`
- **parser** (`benchmark_results.py`): `float(val.strip())`

`float("2395.8 ms  (15 samples)")` raises `ValueError` on **every** metric line,
so the `except ValueError: pass` swallowed all five, the dict stayed empty, and
`if not metrics: raise RuntimeError("no latency metrics parsed")` fired. The
`[OK]` came from `benchmark_latency.py` itself, which is a separate process and
had already decided the run was fine — the caller's parse failure never reached
the user as a failure.

So `record --kind latency` had **never once succeeded**.

# Why tests didn't catch it

`test_benchmark_results.py` covered the stability parser, thresholds, history
rendering and round-trips — but `_run_latency` had **zero** coverage. The
printer and the parser are different files, so nothing pinned them together.

# Fix

`_first_float(text)` — take the leading number, keep the unit on the line.
The printer's human-readable format is the contract; the parser adapts to it,
not the other way round. Changing the printer instead would have broken the
human-readable output and any other consumer for no gain.

# Tests (the point of the change)

The new tests copy `benchmark_latency.py`'s output **verbatim** into a module
constant rather than paraphrasing it, because a paraphrase would have encoded
the same wrong assumption and the drift would return. One test pins the parse,
one pins that a genuinely unparsable run still raises — the guard is kept, only
the false positive is removed.

# Files

- `scripts/benchmark_results.py` — `re` import, `_first_float()`, parse uses it
- `packages/core-py/tests/test_benchmark_results.py` — `_LATENCY_STDOUT`, 2 tests

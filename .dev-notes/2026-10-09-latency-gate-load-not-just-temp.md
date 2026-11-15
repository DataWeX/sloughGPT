---
title: latency baseline guard gated on temperature only — a busy host passed it
date: 2026-10-09
status: done
branch: feat/aml-stream-loader
tags: [benchmark, latency, baseline, host, gate]
---

# Symptom

`scripts/benchmark_latency.py --update` created a baseline, then a later
`record --kind latency` reported `mean 3102.6 → 7919.1 (Δ+4816)` as a
**regression**. It was not a code regression — the serving path had not been
touched by any commit on the branch.

# Root cause

The baseline guard had **one clamp: temperature**. Temperature is a *lagging
proxy* for contention, not the thing itself:

- baseline created at load **0.78**, zone3 66 °C → mean **2295.9 ms**
- measured at load **15.44**, zone3 84 °C → mean **7919.1 ms**

The busy run **passed** the thermal check at 68 °C: the box had been busy for
minutes but had not yet soaked enough to be hot. So the gate approved exactly
the contaminated run it exists to stop — and since the baseline is read by
every later run and by `--ci`, that poisons comparisons permanently, not for
one run.

`_host_context()` was **already reading `load1` from `/proc/loadavg`** and
storing it in the baseline. The data was collected and then ignored by the
gate.

# Fix

- `_load_limit(cpus=None, busy=0.5)` — a fraction of CPU count, so the gate
  scales with the box instead of hard-coding 8 cores. Floor of 1.0 because an
  idle machine never reports load 0.5.
- `_wait_for_thermal()` now evaluates **both** clamps and returns
  `blockers: ["thermal", "load"]` naming which held. Still proceeds on timeout
  with `ok=False` rather than hang — same contract, stricter judgement.
- `_refusal_reason()` names the clamp that actually held. The old message
  asserted "package at NC" unconditionally, so a load-blocked run printed
  **"package at 68C"** — a number that reads as healthy and contradicts the
  refusal attached to it.

# Why this is not a "while I'm here"

The 7919 ms record was written by the very `record --kind latency` run this
thread set out to make work. A parser fix that persists a *contaminated*
number is a fix that hands the next person a wrong baseline. Closing the
thread means the number that lands is trustworthy.

# Findings not fixed here

- **`metrics.model` is `None`** in stored records. `_run_latency` scrapes
  stdout; `benchmark_latency.py` fetches the model from `/health` but never
  prints it. So `do_record` falls back to `"served"` and history entries are
  not attributable — `compare` happily compared `Qwen/...` against `served`.
  Needs a design call on the bridge between the two files (a prior-session
  decision already rules out a `--json-out` bridge).
- **One load-contaminated record** sits in history (mean 7919). Baseline is
  clean — `record` uses `update_baseline=False`.

# Files

- `scripts/benchmark_latency.py` — `_load_limit()`, widened `_wait_for_thermal()`, `_refusal_reason()`, headroom print
- `packages/core-py/tests/test_benchmark_latency.py` — 10 new tests
- `packages/core-py/tests/test_benchmark_results.py` — `_LATENCY_STDOUT` re-copied verbatim for the new printer line

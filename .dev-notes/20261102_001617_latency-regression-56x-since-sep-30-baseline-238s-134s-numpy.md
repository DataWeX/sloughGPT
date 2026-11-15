---
id: 20261102_001617_latency-regression-56x-since-sep-30-baseline-238s-134s-numpy
status: todo
board: 20261102_001617_latency-regression-56x-since-sep-30-baseline-238s-134s-numpy
title: "Latency regression ~5.6x since Sep 30 (baseline 2.38s -> 13.4s numpy path) — pre-dates 2026-11-01 work"
tags: ["perf", "latency", "inference", "regression", "area:inference"]
sprint: ""
gh: null
---

# Latency regression ~5.6x since Sep 30 (baseline 2.38s -> 13.4s numpy path) — pre-dates 2026-11-01 work

Measured 2026-11-01, quiet box, rate-limit-safe harness (7s spacing, 15 samples): numpy default path mean 13.4s / min 10.9s / p95 14.7s vs data/benchmark_latency_baseline.json (Sep 30) mean 2.38s / min 1.66s / p95 2.86s on identical payload (Qwen2.5-0.5B, POST /chat, max_tokens=20, temp=0.01). ~5.6x, survives (a) killing a 3.9h runaway soak that was eating 1 core, (b) quiet-box re-run, (c) both numba-on and numba-off arms (24.5s / 13.4s). So it is NEITHER today's numba install NOR the prefill fallback fix (compiled branch byte-identical in a6c8f3b31). Not yet diagnosed. Leads: swap was fully exhausted 3.8/3.8Gi during session (was it swapping on Sep 30?), baseline ran at pkg_temp 97C with 70s thermal wait (different thermal regime), and a cupy/numpy type-mismatch makes training GPU matmul silently fall back to CPU (1703 lines/10s spam in temp-server logs). Also NOTE scripts/benchmark_latency.py is unusable as-is: /chat is limited to 10 req/60s (apps/api/server/infrastructure/rate_limit_middleware.py), 25 back-to-back requests -> 15x 429, fake 8.8ms min. Use a spaced harness. DoD: bisect the regression between Sep 30 baseline and 2026-11-01 (git log window), record result in benchmark history (scripts/benchmark_results.py), update benchmark_latency.py to space requests or fix the script's rate-limit trap.

---
id: 20260803_041228_qwen-gold-standard-benchmark-stability-fail-latency-baseline
title: Qwen Gold Standard benchmark: stability FAIL + latency baseline
status: done
tags: benchmark,stability,latency,defect
created: 2026-08-03T04:12:28.449609+00:00
---

Qwen Gold Standard benchmark: stability FAIL + latency baseline

Benchmarked served Qwen/Qwen2.5-0.5B-Instruct against Gold Standard + latency baseline.

## Results (recorded via scripts/benchmark_results.py)
- Stability: FAIL. overall=90. length_cv ~0.80 (threshold ≤0.30). Latencies bimodal (~47s real, ~70s = 60s timeout placeholder). 20 runs, ~23 min.
- Latency baseline: mean 31.7s, p50 28.0s, p95 57.8s (15 samples, fresh session per request). Docs claim ~2s — 15x slower than documented on this CPU box.

## Work done
- scripts/benchmark_latency.py: fixed dead /labs/chat endpoint → POST /chat; client timeout 30s → 120s (server takes 47-70s); unique session_id per request to avoid SessionKVCache replay (was poisoning results with ~0.017s cached responses).
- scripts/benchmark_stability.py: _resolve_model unwraps health envelope (data.model_type); unique session_id per request.
- scripts/benchmark_results.py: NEW harness — record/history/compare for stability + latency runs; direction-aware regression thresholds; tests in packages/core-py/tests/test_benchmark_results.py (15 pass).

## Defects found
1. 60s ChatDomain timeout (packages/core-py/domains/chat/domain.py:150) fires on most requests → server returns exactly '[Error: Generation timed out after 60 seconds]' (46 chars) as HTTP 200. Model cannot finish 50 tokens in 60s on this box.
2. Qwen output degenerate/rambling — ignores 'Say hi in 3 words', emits '<|im_start|>/<|im_end|>' boilerplate. Not Gold Standard quality.
3. SessionKVCache (model_server.py:299) stores generated sequence per session — same prompt + same session replays cached tokens in ~0.017s, corrupting benchmarks unless fresh session_id used.
4. SloNet generate timeout is 70s (slonet_server.py:334) vs documented 120s; ChatDomain 60s wait_for fires first.

## Files
- data/benchmark_results/stability/Qwen--Qwen2.5-0.5B-Instruct_2026-08-03T035353166894+0000.json (FAILED)
- data/benchmark_results/latency/Qwen--Qwen2.5-0.5B-Instruct_2026-08-03T041201363061+0000.json
- data/benchmark_latency_baseline.json (mean 31.7s)

## Next steps
- Decide whether to re-run stability with fresh sessions (clean Gold Standard verdict) and/or fix the 60s timeout / degenerate generation defects.
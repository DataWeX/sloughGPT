---
id: 20260929_084
board: a25e490b-6d8e-4d05-9a20-ba29ffcc487e
title: "Card 084 — Tokens/sec investigation vs reported slonet 30 tok/s"
status: done
date: 2026-09-29
tags: [perf, inference]
---

# Tokens/sec investigation vs reported slonet 30 tok/s

## Background

Ask 2026-08-28: 'we are still having issues with our tokens per sec, I hear slonet is 30 per sec'. Run scripts/benchmark_slonet.py + benchmark_latency.py, profile decode path, quantify gap, decide fix -> implementation card. Verify no existing card covers this first.

## Source
Mined from stale opencode sessions (Aug 26-31 2026) at user's request: ~/.cache/slog-journeys/session-exports/.

## Findings (2026-09-29)

**Measured decode: 8.6 tok/s (in-process) / 6.6 tok/s (e2e /chat). Never 30.**

### 1. Core suite — `scripts/benchmark_slonet.py` 7/7 PASS
Tiny SloTransformer generation (8+20 tok): 102ms (~274 tok/s) — core kernels
fine at micro scale; the gap is in the real-model decode path.

### 2. In-process Qwen2.5-0.5B int8 (quantize=True, symmetric, AVX2)
- load 9.1s, warmup gen(32) 2.7s
- gen 64 tok: 5.84s (11.0 tok/s naive); gen 256 tok: 28.07s (9.1 tok/s naive)
- **slope decode = 8.63 tok/s** (116 ms/token)
- cProfile (128 tok, 14.83s):
  - `quant_core/wrapper.py:269 matmul_int8_f32_c` — **10.74s = 72.5% of all CPU**
    (12544 calls = 98/token, 0.86ms/call) ← THE bottleneck
  - `_sample_from_logits` 0.41s (3.2ms/token)
  - quantize_kv 0.54s + dequantize_kv 0.13s, nb_rmsnorm 0.51s,
    fused_attention 0.46s, swi_glu 0.15s, c_einsum 0.23s
- Bandwidth math: ~494M int8 weights ≈ 494MB read/token; measured effective
  4.3GB/s → 116ms/token. A normal DDR path (25–40GB/s) floors at 12–20ms/token
  → **ceiling 50–80 tok/s. Kernel runs at ~14% of memory bandwidth.**

### 3. E2E — `scripts/benchmark_latency.py` vs live server (port 8000)
- pass 1: mean 3014.8ms / 20 tok = 6.6 tok/s; pass 2: 3018.1ms (stable)
- reported "+145.3% vs baseline" is **mostly a measurement artifact**:
  baseline (2026-08-20) recorded `max_tokens: 10` (mean 1228.8ms = 8.1 tok/s,
  123ms/tok); current script sends 20 → per-token real regression = +23%
  (123 → 151ms tok), not +145%.

### 4. The "30 tok/s" claim
Only 30-tok/s claim in repo = `docs/engineering-overview.md:129` — that's
**llama-rn (Metal, 15-30 tok/s)**, a different engine. Slonet CPU int8 has
never measured ≥30 tok/s here (Aug-20 e2e was 8.1 tok/s).

### Outcome
→ Fix card: optimize `matmul_int8_f32_c` GEMV path (target ≥25 tok/s decode =
≥10GB/s effective) + secondary sampling/KV-quant trims + re-pin latency
baseline with matching max_tokens.

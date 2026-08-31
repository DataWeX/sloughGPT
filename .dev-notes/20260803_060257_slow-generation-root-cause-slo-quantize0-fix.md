---
id: 20260803_060257_slow-generation-root-cause-slo-quantize0-fix
title: Slow generation root cause + SLO_QUANTIZE=0 fix
status: done
tags: inference,quantization,perf
created: 2026-08-03T06:02:57.308131+00:00
---

Slow generation root cause + SLO_QUANTIZE=0 fix

Root cause of slow generation (1.89 tok/s): int8 quantization used the numpy fallback because the AVX2 C extension .so is not built on this Linux machine. Micro-benchmark: quantized_linear 14.4ms vs plain float32 1.15ms = 12.6x slower. Fixed by launching server5 with SLO_QUANTIZE=0 (quantize_slonet=False -> SloLinear.forward_numpy uses x @ weight.T float32 BLAS). Throughput: 5.6-7.5 tok/s (3.7x). Verified captured-conversation status live: /mobile/train/auto-status shows captured_count=16 (5 prior + 11 benchmark rows). Corpus: datasets/api_conversations/corpus.jsonl (16 rows) + input.txt (71 lines). The 30+ tok/s seen before was on macOS where quant_core matmul_int8.so was compiled.
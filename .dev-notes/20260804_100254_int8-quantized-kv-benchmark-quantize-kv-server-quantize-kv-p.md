---
id: 20260804_100254_int8-quantized-kv-benchmark-quantize-kv-server-quantize-kv-p
title: int8 quantized KV benchmark (--quantize-kv) + server quantize_kv param
status: done
tags: kv-cache,quantization,benchmark,slonet
created: 2026-08-04T10:02:54.297945+00:00
---

int8 quantized KV benchmark (--quantize-kv) + server quantize_kv param

Added --quantize-kv to scripts/benchmark_kv_reuse.py across all three modes (direct, --stack, --stack --stream). SloNetServer gained a quantize_kv constructor param forwarded to both generate_numpy and generate_numpy_stream. All three int8 modes: reuse 0->12->23->34 (identical to float32, prefix-match quantity), 100% warm/cold consistency, exit 0. tests/test_benchmark_kv_reuse.py 20->27 tests (+7: int8 reuse growth + consistency for all modes + float-vs-int8 reuse equality). QUANTIZATION_BENCHMARK_PLAN.md updated. Regression suites green (slonet_server, session_ttl, kv_benchmark, chat_domain).
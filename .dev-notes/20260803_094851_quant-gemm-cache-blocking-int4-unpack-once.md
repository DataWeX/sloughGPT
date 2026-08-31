---
id: 20260803_094851_quant-gemm-cache-blocking-int4-unpack-once
title: Quant GEMM cache-blocking + int4 unpack-once
status: done
tags: quant,kernels,perf
created: 2026-08-03T09:48:51.400821+00:00
---

Quant GEMM cache-blocking + int4 unpack-once

Blocked B j-loop (256KB) in matmul_int8.c so weights stream from DRAM once; int4 kernels unpack each B row once per j-block into int8 scratch for M>1, inline for M=1. Benchmark test_int4_vs_int8_speed now passes (was 2.15x fail; now 0.79x at M=128, 1.08x lm_head). Correctness verified vs numpy. 101 passed, 3 skipped across quantization suites.
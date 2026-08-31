---
id: 20260803_100218_wrapper-stale-so-rebuild-fix-native-blocking-tests
title: Wrapper stale-.so rebuild fix + native blocking tests
status: done
tags: 
created: 2026-08-03T10:02:18.798537+00:00
---

Wrapper stale-.so rebuild fix + native blocking tests

Quant kernel hardening done. (1) wrapper._build_all now rebuilds when C source is newer than .so (mtime) + force= param; tests cover rebuild/skip/force. (2) Added TestNativeBlocking: multi-j-block N, M=128 row reuse, odd/scalar-remainder dims, int4 M=1 inline + M>1 scratch paths — all verified against fallback. (3) Investigated int4 M=1 (13ms) vs int8 M=1 (12ms) lm_head: attempted VPMADDUBSW fold (a·b=a·u-8·Σa) but packed int4 stores 2 nibbles/byte, whole-byte maddubs cannot multiply nibbles — unpack is a hard per-element compute floor (~0.66 ops/elem vs int8 0.375), so M=1 int4 is compute-bound ~= int8. Reverted; int4's win is the M>1 scratch path (benchmark 0.79x). Suites green: quant_core(35)+quantization+benchmark+provider(101,3 skip)+coverage(31). E2E Qwen quantized gen unchanged.
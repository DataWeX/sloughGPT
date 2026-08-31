---
id: 20260803_113931_threaded-int8-gemm-j-block-loop-opt-in-via-man-gemm-threads
title: Threaded int8 GEMM j-block loop (opt-in via MAN_GEMM_THREADS)
status: done
tags: sloughgpt,quantization,inference,optimization
created: 2026-08-03T11:39:31.167280+00:00
---

Threaded int8 GEMM j-block loop (opt-in via MAN_GEMM_THREADS)

Threaded the AVX2 int8 GEMM j-block loop (pthread): per-thread contiguous j-block column slices keep output bit-identical to serial. -pthread added to _build_one; MAN_GEMM_THREADS=N (1..256) env knob. Bit-identity verified via TestThreading (4 new tests in test_quant_core.py; 41 quant_core tests pass). 6MB (6291456B) min-B-bytes threshold keeps small GEMMs serial.

Measurements on this 4C/8T i5-9300H (load ~1.7, noisy):
- Isolated interleaved medians: lm_head (151936,896) 32.2->13.0ms (2.48x); ff_fused (9728,896) 2.20->1.48ms (1.49x); w2/o (4864,896) 0.44->0.46 (neutral, below threshold); qkv (1152,896) neutral.
- Real warm decode (fresh-process medians): 1th 58.2, 2th 57.3, 4th 57.1, 8th 63.9 ms/token. 8 threads regressed ~10% (HT contention + heat throttle); 2-4 threads ~1% (noise level). C-time profile (in-proc, patched quant_linear): ~47-57ms/token of the ~54-58ms decode is C GEMM (86%).
- Key finding: isolated big-GEMM gains do NOT transfer to real decode (warm weights already near DRAM BW; threading adds contention and stalls numpy attention ops). At 4 threads decode was bimodal (59ms vs 124-130ms) while serial stayed stable 54-67ms.

Decision: threading is now OPT-IN (serial by default). _gemm_threads() returns 1 unless MAN_GEMM_THREADS is set. Keeps bit-identical output and a tuning knob for beefier multi-socket machines; avoids the throttle/contention regression as default.
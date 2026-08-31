---
id: 20260804_023413_asymmetric-int4-vectorization-38x-packed-int4-fused-qkvffn-g
title: Asymmetric int4 vectorization (38x) + packed int4 fused QKV/FFN generation path
status: done
tags: quantization,inference
created: 2026-08-04T02:34:13.016782+00:00
---

Asymmetric int4 vectorization (38x) + packed int4 fused QKV/FFN generation path

Packed int4 fused path for generate_numpy/generate_numpy_stream in slonet.py.

Changes:
- New _fuse_quant_weights_int4(): builds packed (N, K//2) fused W/S/zp/bias when every layer in a group is int4 with matching zero point and even input dims. Keeps int4 memory compression end-to-end.
- _fuse_quant_weights() now returns None when any layer has a non-zero zero point (fused int8 path hardcodes zero_point=0, which silently produced wrong logits for asymmetric int8).
- generate_numpy() and generate_numpy_stream() prefer the packed int4 fusion (QKV and FFN); the int8 fusion is only built per-block when the int4 fusion is unavailable (previously eager int8 build forced int4 layers through _get_quant_array() and unpacked them).
- Vectorized the asymmetric int4 unpack loop in int4_matmul via _unpack_int4: 489.8 ms -> 12.9 ms at 128x768x768 (~38x), byte-identical layout.

Verification:
- tests/test_quantization_integration.py::TestGenerateNumpyPackedInt4 (5 tests): packed fused output byte-identical to per-layer path; no _quant_unpacked materialized; stream matches generate_numpy; int8 symmetric fuse still active; asymmetric int8 falls back.
- Full default suite: 10729 passed, 38 skipped, 0 failures.
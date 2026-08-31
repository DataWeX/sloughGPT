---
id: 20260803_014010_wave-g-gpu-accelerator-coverage-26-96
title: Wave G: GPU accelerator coverage (26% -> 96%)
status: done
tags: training,coverage,gpu
created: 2026-08-03T01:40:10.780603+00:00
---

Wave G: GPU accelerator coverage (26% -> 96%)

Completed: wave G — GPU accelerator layer coverage 26% -> 96%.

New tests/test_gpu_accelerator.py (68 tests): every compute op of _CPUAccelerator, _MetalAccelerator, and the cupy-fallback path of _CUDAAccelerator verified against independent numpy references (softmax, gelu, silu, layernorm, rmsnorm, attention, scaled-dot-attention with mask/scale variants, dropout, embedding incl. clipping/OOR, cross_entropy incl. OOR-skip, conv2d, max_pool2d); CUDA env detection (CUDA_VISIBLE_DEVICES branches); global get_accelerator/to_gpu/from_gpu/reset (cache + CPU selection); Cholesky solvers vs np.linalg; dominant_eigen power iteration vs eigvalsh.

Bugs fixed in accelerator.py:
- _CPUAccelerator.conv2d required bias positionally (inconsistent with Metal/CUDA) — now defaults to None.
- _MetalAccelerator.max_pool2d crashed on tuple strides ('//' between int and tuple) — strides now split into sh/sw.

Environmental floor (8 stmts, all cupy/MPS-only): cupy import/dispatch arms (lines 164-165, 179, 184, 189, 194) and CUDA/Metal selection in get_accelerator (375, 377). Pure delegation to a third-party GPU lib with no in-repo algorithm to verify.

Full wave suite (16 files): 733 passed. slonet.py 97% (unchanged, 3 numba-probe misses), slonet_kernels.py 99%, accelerator.py 96%. TOTAL 4517 stmts / 11 miss / 98%.
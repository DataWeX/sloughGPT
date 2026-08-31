---
id: 20260731_053037_fix-slonet-mul-backward-broadcast-sensitivity-test
title: Fix SloNet mul backward broadcast + sensitivity test
status: done
tags: slonet,autograd,bugfix
created: 2026-07-31T05:30:37.953000+00:00
---

Fix SloNet mul backward broadcast + sensitivity test

Fixed pre-existing SloNet failures found during the soul.py regression run.

- slonet.py _mul backward: replaced buggy manual sum loop (only reduced size-1 dims, ignored leading extra dims for lower-ndim inputs, misaligned positions) with _broadcast_back, the same helper _add already uses. Fixes 6 failing test_slonet_broadcast.py tests (e.g. grad shape (3,4) vs (4,) for a=(4,) * b=(3,4)). 13/13 pass.
- test_slonet_compute_sensitivity.py: np.isfinite(p.grad) -> p.grad.data (contract is .grad is a Tensor; file inconsistent with its own line 115). 9/9 pass.
- Verified no regressions: 170 slonet-dependent tests pass (distillation, forward_pass, fused_kernels, lora, lr_schedulers, point_weight, pugqeep_checkpoint, quantization_integration, quantized_matmul), plus bidirectional_dag/integration/soul_engine. py_compile OK, pycache cleared.
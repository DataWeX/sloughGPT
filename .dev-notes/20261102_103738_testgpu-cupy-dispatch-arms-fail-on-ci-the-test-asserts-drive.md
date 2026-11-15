---
id: 20261102_103738_testgpu-cupy-dispatch-arms-fail-on-ci-the-test-asserts-drive
status: done
board: 20261102_103738_testgpu-cupy-dispatch-arms-fail-on-ci-the-test-asserts-drive
title: "test(gpu): cupy dispatch arms fail on CI — the test asserts driver presence nobody guarantees"
tags: ["testing", "gpu", "ci"]
sprint: ""
gh: null
---

# test(gpu): cupy dispatch arms fail on CI — the test asserts driver presence nobody guarantees

FOUND 2026-11-02 ~10:30 in CI/CD run 41422832686 (8ff6f3432), job test-core-py.

5 reds, all TestCudaWithFakeCupy: test_cuda_available_with_fake_cupy,
test_cuda_dtype_fp32_and_fp16, test_cuda_to_from_device_with_fake_cupy,
test_cuda_vram_full_tier, test_cuda_vram_lite_tier.

THE BUG IS THE TEST, NOT THE MODEL. `_CUDABackend.is_available()` first calls
`_cuda_driver_present()` (dlopen libcuda.so.1) and returns False before it ever
imports cupy. GitHub runners have no libcuda, so the fake-cupy module installed by
monkeypatch.setitem(sys.modules, 'cupy', ...) is never reached: is_available() is
False, _cp stays None (-> AttributeError: 'NoneType' has no attribute float32), and
vram_gb() falls back to 4.0 (-> tier 'lite' vs asserted 'full').

Proof of the environment dependency: the same 183 tests pass locally, because this
dev box HAS libcuda.so.1 installed (no GPU needed for dlopen to succeed) while CI
does not. So the suite is green on the machine that writes it and red on the machine
that gates it — the worst split.

FIX: the class tests the cupy dispatch arm, not driver probing, so pin the seam:
a class-level autouse fixture monkeypatching slib._cuda_driver_present -> True.
Driver probing keeps its own coverage (test_slolib_gpu has direct probes).

WHY IT BLOCKS: CI/CD has been red on main for every push in this window, so the
deploy half of the pipeline cannot go green and real regressions hide behind it.

---
id: 20260803_025224_gpu-coverage-100-metalopenclpsutil-dispatch-arms
title: GPU coverage 100%: Metal/OpenCL/psutil dispatch arms
status: done
tags: gpu,slolib,test-coverage
created: 2026-08-03T02:52:24.614897+00:00
---

GPU coverage 100%: Metal/OpenCL/psutil dispatch arms

Verified uncommitted GPU bug-fix work (7 fixes: _BufferPool dtype normalization, conv2d pad+channel-order, flash-attention causal k, CUDA mem_info/triu/gelu) and extended tests/test_slolib_gpu.py to 100% statement coverage of domains/slolib/gpu/__init__.py (779/779, up from 88%). Added TestMetalWithFakeTorch (fake torch proxy: tensor/nn.functional ops), TestOpenCLWithFakeOpenCL (fake pyopencl: is_available/vram_gb tiers/to-from device/lazy context import), psutil-path tests in TestDeviceBasics, and device-transfer pass-through tests. Fixed fp16 matmul tolerance to 1e-2 (input-rounding semantics) and OpenCL medium-tier max_seq_len 256. Regression green: test_slolib_gpu (all), test_slonet_legacy, test_gpu_accelerator, test_gpu_precision, test_slonet_wave_a, test_tokenizer.
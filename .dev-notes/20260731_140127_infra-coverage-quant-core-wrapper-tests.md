---
id: 20260731_140127_infra-coverage-quant-core-wrapper-tests
title: Infra coverage: quant_core wrapper tests
status: done
tags: infrastructure,tests
created: 2026-07-31T14:01:27.995489+00:00
---

Infra coverage: quant_core wrapper tests

Added tests/test_quant_core.py (14 pass, 5 skipif-native) for quant_core/wrapper.py - last untested infra module with production consumers (quantization.py, routers/models.py import HAS_AVX2/matmul_int8_c/matmul_int4_c). Covers: numpy int8/int4 fallback vs int32 reference, forced-fallback via monkeypatched _load_lib, int4 pack/unpack round trip, _build_one failure modes (missing src / gcc error / gcc absent), _build_all with and without prebuilt libs, _load_lib idempotence. Environment has no gcc -> HAS_AVX2=False, native C paths skipif. Also: deployment/__init__.py confirmed dead code (0 consumers repo-wide, imports OK) - flagged for deletion, not tested (pt_loader precedent). Full slnc (80) + quant_core (14) + quantization suites green; pre-existing test_safetensors_loader env failures unchanged (need real gpt2 + safetensors pkg).
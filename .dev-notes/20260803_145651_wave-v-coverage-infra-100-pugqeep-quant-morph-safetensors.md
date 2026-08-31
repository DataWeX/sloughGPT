---
id: 20260803_145651_wave-v-coverage-infra-100-pugqeep-quant-morph-safetensors
title: Wave V+ coverage: infra 100% (pugqeep, quant, morph, safetensors)
status: done
tags: infrastructure,coverage,core-py
created: 2026-08-03T14:56:51.486365+00:00
---

Wave V+ coverage: infra 100% (pugqeep, quant, morph, safetensors)

Completed Wave V+ coverage push for domains/infrastructure. All target modules at 100% test coverage via 694 passing tests (17 skipped).

Modules at 100%:
- morph_tokenizer.py, safetensors_loader.py (Wave V)
- pugqeep/* (13 modules): point, store, library, compressor, dedup, model_tree, queue, facade, cache, generic, config, task_queue, point_weight, __init__
- quant_core/wrapper.py (81%->100%): added real-file _build_one tests (success, gcc failure, gcc missing, generic exception), _build_all OSError-on-mtime, _load_lib no-library path, matmul_int8_f32_c missing-symbol, and importlib.reload platform tests (darwin .dylib, win32 .dll, int8-only load, numpy-fallback import)
- quantization.py (95%->100%): full 6-file quant suite covers remaining old int8/quantize_weights branches

Verification: /tmp/cov_final aggregate (16 infra test files + test_morph_tokenizer_wave_g.py); 647+47 passed, 17 skipped; py_compile clean; pycache cleared.

Blocked (unchanged): tests/test_rate_limiter.py (starlette not installed); tests/test_lifecycle.py::test_lifecycle_endpoint (ModuleNotFoundError).
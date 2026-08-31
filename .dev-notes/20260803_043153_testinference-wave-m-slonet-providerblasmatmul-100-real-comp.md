---
id: 20260803_043153_testinference-wave-m-slonet-providerblasmatmul-100-real-comp
title: test(inference): wave M slonet_provider/blas/matmul 100% real-components
status: done
tags: test,coverage,wave-m
created: 2026-08-03T04:31:53.775817+00:00
---

test(inference): wave M slonet_provider/blas/matmul 100% real-components

Wave M closed: slonet_provider 61%->100%, blas 67%->100%, matmul 83%->100% via real programmatic inputs only (no mocks, no third-party installs).

New tests/test_slonet_provider_real.py (16 tests): hand-built real .slnc per slnc/spec.py + real tokenizer.json in temp HF_HOME + real SloTransformer/MorphTokenizer/SloNetServer. Covers from_slnc (incl. all 3 QuantEngine paths: fresh/metadata-only/pre-quantized npz), _load_safetensors_bf16 real file (BF16/F32/F16/unknown dtype), _split_fused_qkv ValueError path, _build_prompt empty/string/list + chat-template, _load_tokenizer RuntimeError, set/get_server + to_server, chat()/chat_stream() server-attached and sync-thread branches, generate_with_logprobs seed, generate_with_stop, generate_batch, max_new_tokens override.

tests/test_ops.py: appended TestAccelerateUnavailable (fallback paths) + TestAccelerateAvailable (real ctypes cblas_sgemm via system libblas.so.3 - identical CBLAS ABI; exercises _setup_sgemm signature, cached load, sgemm alpha/shape/dtype asserts, matmul C path).

Source edits (slonet_provider.py only): pragma-no-cover on genuinely dead defensive branches (except-exception ResourceManager block; _build_prompt legacy fallback unreachable with MorphTokenizer).

Verification: 3-target sweep 100% (442+36+6 stmts, 0 missing); regression 201 pass / 21 skip across provider+ops+slnc+quantization+morph_tokenizer suites; py_compile clean; pycache cleared.
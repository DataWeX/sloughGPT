---
id: 20260803_013212_wave-h-slnc-specparsercompiler-resource-manager-100-coverage
title: Wave H: slnc spec/parser/compiler + resource_manager 100% coverage
status: done
tags: slonet,coverage,slnc,resource-manager
created: 2026-08-03T01:32:12.357915+00:00
---

Wave H: slnc spec/parser/compiler + resource_manager 100% coverage

Wave H: pushed slnc/* + resource_manager.py to 100% coverage.

spec.py: fixed latent bug — code_to_dtype(2) raised AttributeError (numpy 2.5.1 has no np.bfloat16); added hasattr guard raising ValueError. New tests/test_slnc_spec.py (5 classes, parametrized): spec.py 50/50.

resource_manager.py: added 5 tests to tests/test_resource_manager.py (numexpr fake-module injection, numexpr-missing silent, singleton-None lazy init, np compute limits, singleton global) — 163/163.

compiler.py + parser.py: extended tests/test_slnc_compiler.py with BF16/F16/unknown-dtype safetensors load branches, default output path (output=None -> models dir, chmod-444 artifact cleaned), protect_model failure silence, parser edges (unsupported version, file_size, exception-safe __del__, repr, get_weights_dict), LLaMA bias auto-append, _xxhash64 installed-path via fake xxhash module. compiler.py 172/172, parser.py 107/107.

Also fixed flaky Wave F test: test_slonet_wave_f.py bf16 assertion used random data under atol=1e-2 -> deterministic bf16-exact values.

Combined: 492/492 stmts (100%) across slnc/* + resource_manager. 8 consecutive clean runs; full slonet family (slnc + resource_manager + kernels + morph tokenizer + wave_g) green.

Note: gpu/accelerator.py Metal stride-tuple fix + test_gpu_accelerator.py remain uncommitted (pre-existing orphaned WIP, excluded from this commit).
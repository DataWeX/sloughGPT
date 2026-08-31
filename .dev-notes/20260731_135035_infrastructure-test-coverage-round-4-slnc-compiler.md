---
id: 20260731_135035_infrastructure-test-coverage-round-4-slnc-compiler
title: Infrastructure test coverage round 4: slnc compiler
status: done
tags: infrastructure,tests
created: 2026-07-31T13:50:35.835760+00:00
---

Infrastructure test coverage round 4: slnc compiler

Round 4: 26 tests for domains/infrastructure/slnc/compiler.py (last uncovered infra module) in tests/test_slnc_compiler.py. Round-trip compile->SLNCParser for GPT-2 and LLaMA (incl. lm_head weight tying), header metadata, tensor table layout (offsets/entries/crc/ordering), 64-byte data alignment, CRC verify_all + corruption detection, invalid magic rejection, HF safetensors->slnc path via monkeypatched safetensors_loader. Found+fixed real bug: _order_tensors duplicated GPT-2 biases (biases are in GPT2_BLOCK_LAYOUT AND auto-appended by the .weight->.bias check) -> tensor_count 22 vs 16, duplicated bias bytes on disk; auto-append now skips biases already in block layout (LLaMA unaffected). 80/80 slnc tests green (format+loader+compiler). xxhash fallback path verified (module not installed).
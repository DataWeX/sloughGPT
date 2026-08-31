---
id: 20260816_132602_test-suite-fix-torch-removal-test-compatibility
title: Test suite fix: torch removal test compatibility
status: done
tags: test,torch,status:done
created: 2026-08-16T13:26:02.167350+00:00
---

Test suite fix: torch removal test compatibility

Fixed torch removal test compatibility properly.

Root cause: onnx_export.py and hf_finetune.py stubs raised ImportError at import time, crashing any test file that imported from them.

Changes:
1. onnx_export.py — Rewrote as proper stub module: ONNXExportConfig is a pure dataclass (works without torch), SloughGPTONNXExport and export_sloughgpt_to_onnx raise ImportError when CALLED (not at import). _TORCH_AVAILABLE = False.
2. hf_finetune.py — Same pattern: HFFineTuner and TextFileDataset raise ImportError when CALLED, merge_lora_adapter checks for peft first. _HF_FINETUNE_AVAILABLE = False.
3. test_onnx_export.py — Rewrote: tests ONNXExportConfig (pure data), verifies _TORCH_AVAILABLE is False, verifies stubs raise ImportError when called.
4. test_hf_finetune.py — Rewrote: tests _HF_FINETUNE_AVAILABLE, HFFineTuner/TextFileDataset/merge_lora_adapter stub behavior.
5. test_export.py — Removed 5 skipped test classes (tested deleted ONNX/safetensors functions), removed unused fixtures/helpers (fake_torch, fake_safetensors, stub_exports, _cfg, _FakeTorch, etc.), cleaned imports. File went from 872 lines to ~270 lines.

Results: 1090 passed, 39 skipped, 2 warnings. All compile checks pass.
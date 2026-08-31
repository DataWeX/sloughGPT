---
id: 20260801_060510_torch-removal-from-export-layer-slonet-compat-deletion
title: Torch removal from export layer + slonet_compat deletion
status: done
tags: core,torch,export,slonet
created: 2026-08-01T06:05:10.410068+00:00
---

Torch removal from export layer + slonet_compat deletion

Completed torch-free migration of the export layer and removed the torch shim.

- gguf_export.py: dropped slonet_compat; numpy-first conversion via _as_float16 (handles numpy/SloNet/torch tensors); annotations use StateDict/ModelLike aliases. Verified: arch detection, 21-key tensor map, 12 quants, optional gguf package guard.
- onnx_export.py: import-safe without torch; torch mirror classes + export_sloughgpt_to_onnx guarded behind optional import, clear ImportError when torch missing.
- Deleted slonet_compat.py (1936 lines), hf_dpo.py (378), migrate_torch.py (184). Zero live module-level consumers (all API/test refs lazy or importorskip-gated).
- test_optimized_pipeline.py:139: slonet_compat alias -> plain torch.
- pytest.ini/docs/pyproject clean; torch absent from declared deps.
- Suite: 5535 passed, 16 pre-existing env/algorithm failures (fastapi missing, numpy_forward matmul dim bug, hub/network, task queue, wandb) - none in touched domains.
- All remaining torch imports repo-wide are lazy/guarded feature imports (ONNX/TorchScript export, HF fine-tune, MPS detection, NPU kernels, CLI commands); module imports stay torch-free.
- AGENTS.md updated: slonet_compat removed from architecture docs.

Commit: bfebd3b (pushed to origin/main).
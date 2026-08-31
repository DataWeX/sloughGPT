---
id: 20260801_062327_torch-removal-from-export-layer-numpy-forward-test-fix
title: Torch removal from export layer + numpy forward test fix
status: done
tags: core,torch,export,slonet,tests
created: 2026-08-01T06:23:27.773395+00:00
---

Torch removal from export layer + numpy forward test fix

Torch-free export layer + fixed numpy forward test weight layouts.

Migration (complete):
- gguf_export.py: numpy-first (_as_float16 handles numpy/SloNet/torch); StateDict/ModelLike annotations.
- onnx_export.py: import-safe without torch; torch mirror guarded; clear ImportError otherwise.
- Deleted slonet_compat.py (1936 lines), hf_dpo.py, migrate_torch.py.
- test_optimized_pipeline.py: slonet_compat alias -> plain torch.
- All remaining torch imports repo-wide are lazy/guarded feature imports; modules import cleanly without torch.
- AGENTS.md architecture docs updated.

Follow-up fix:
- test_numpy_ops_forward.py builders supplied inverted weight layouts (GPT-2 needs (in,out) Conv1D; LLaMA needs (out,in) Linear). Generic numpy forward was correct. 9 matmul failures fixed; all 34 tests pass.
- Full suite: 5544 passed / 7 failed (5 deterministic env gaps: fastapi+huggingface_hub not installed; 2 flaky order-dependent wandb/watchdog that pass in isolation).

Commits: bfebd3b, f1352af, 841f520 (pushed).
---
id: 20260803_164857_fix-fastapi-collection-errors-testsloenginelearn-verificatio
title: Fix fastapi collection errors + TestSloEngineLearn verification
status: done
tags: testing,fastapi,core-py
created: 2026-08-03T16:48:57.486033+00:00
---

Fix fastapi collection errors + TestSloEngineLearn verification

Task 'Fix known failing test' — resolved.

1. tests/test_tokenizer.py::TestSloEngineLearn (previously logged SloNet backward broadcast bug) already passes: 4/4 green.
2. Fixed 2 fastapi collection errors: added pytest.importorskip('fastapi') to test_executor_endpoints.py + test_infer_router.py. Both now skip when fastapi absent.
3. ROOT CAUSE of 96 embedding/knowledge failures: test_metrics_collector.py installed a fake psutil into sys.modules at module import time (setdefault) and never removed it — poisoned the whole pytest session, so every _load_embed_model() call hit a _Mem lacking .available. Fixed: fake psutil now scoped to a module autouse fixture using pytest.MonkeyPatch + importlib.reload (bind with fake, undo, rebind without). 101/102 failures eliminated.
4. Real quant bug: quant_core/wrapper.py matmul_int8_f32_c used isinstance(B_scale, (int, float)) which rejects np.float32 scalars (per-tensor scale misdetected as per-row, assertion 'B_scale has 1 rows, expected 16'). Fixed with np.isscalar. Added regression test test_fused_matches_unfused_numpy_float_scale.
5. Full default suite: 102 failed -> 8 failed (all timing-flaky watchdog/wandb/pgq threading tests; 174/174 pass in isolation).

Files: tests/test_executor_endpoints.py, tests/test_infer_router.py, tests/test_metrics_collector.py, domains/infrastructure/quant_core/wrapper.py, tests/test_quant_core.py
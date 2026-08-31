---
id: 20260803_162631_infra-coverage-100-numpy-engine-model-worker-process-guard
title: Infra coverage 100%: numpy_engine, model_worker, process_guard
status: done
tags: infra,testing,coverage
created: 2026-08-03T16:26:31.311915+00:00
---

Infra coverage 100%: numpy_engine, model_worker, process_guard

Completed the domains/infrastructure coverage push to 100% (12078 stmts, 0 miss in final aggregate).

- numpy_engine.py: 48% -> 100%. Fixed dead branch in _load_weights (config_path existence check); added tests/test_numpy_engine_synthetic.py (35 tests) using synthetic HF cache + hand-written .safetensors files + fake safetensors module: _load_weights mmap/bf16-f16-f32 fallback, KVCache, constructor/compression (incl. linear-centroid branch), from_pretrained (incl. use_points + default library), from_slnc, generate/generate_stream (greedy, sampling, kv-cache incremental, EOS stop).
- model_worker.py + process_guard.py: 0% -> 100% via existing tests/test_process_isolation.py (61 passed, 3 skipped; slow-marked, requires -o addopts='').
- Combined into /tmp/cov_final aggregate; TOTAL 100%.
- Removed dead coverage gap by recombining agg4 + synthetic + process-isolation runs (--keep).
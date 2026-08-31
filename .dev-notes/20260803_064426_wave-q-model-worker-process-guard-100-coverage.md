---
id: 20260803_064426_wave-q-model-worker-process-guard-100-coverage
title: Wave Q: model_worker + process_guard 100% coverage
status: done
tags: coverage,infrastructure,process-isolation,test-coverage
created: 2026-08-03T06:44:26.291134+00:00
---

Wave Q: model_worker + process_guard 100% coverage

model_worker.py 42%->100% (207/207) and process_guard.py 78%->100% (129/129) via tests/test_process_isolation.py. Pragma no-cover on 3 child-process-only worker functions (_worker_loop/_slo_worker_main/_hf_worker_main, recorded in parent). Added parent-side tests: spawn-failure fakes (_dead/_silent/_silent_alive worker mains, fake psutil in sys.modules), fake queues/procs, start failure/retry, stop edges, cleanup error swallow, health-check hb drain, generate/stream error paths, monitor-loop callback crash survival, memory_mb, factory config asserts. Final: 54 passed/3 skipped; combined TOTAL 336, 0 Miss; py_compile clean.
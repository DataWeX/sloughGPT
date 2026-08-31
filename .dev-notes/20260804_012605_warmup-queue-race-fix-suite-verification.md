---
id: 20260804_012605_warmup-queue-race-fix-suite-verification
title: Warmup queue race fix + suite verification
status: done
tags: tests,infrastructure,model-server
created: 2026-08-04T01:26:05.650309+00:00
---

Warmup queue race fix + suite verification

Root-caused the sporadic test_registry_metrics_track_generation CancelledError: warmup thread ran self.generate() (-> PriorityRequestQueue.worker()) on a throwaway asyncio.new_event_loop(); the queue stayed bound to that closed loop, and the rebuild path in _ensure_queue() called q.close(), cancelling a live in-flight future.

Fix (model_server.py):
- _run_warmup() rewritten (~line 1116) to bypass the async queue entirely: direct synchronous _generate_sync(...); records metrics itself under _metrics_lock (requests_total, record_success/record_failure); on failure sets _warmup_error + calls _on_generation_error() so status degrades (test_warmup_graceful_on_failure requires DEGRADED).
- Fix swap/compile race: capture ref = self._model_ref, compile, then under self._lock only assign if self._model_ref is ref (syncs with swap_model()); _local_backend._model_ref updated together (fixes test_swap_model 'is' identity assert).
- PriorityRequestQueue.close() + inspect.iscoroutine close added; _ensure_queue() rebuild calls q.close(); swap_model() closes queue + resets _request_queue/_queue_task/_queue_loop.

Verification:
- tests/test_priority_queue.py + test_server_integration.py: 80 passed, 10/10 runs clean, no 'Task was destroyed but it is pending!' warnings.
- test_model_server_units + test_parallel_execution: 132 passed. test_process_isolation + test_server_integration + test_priority_queue: 146 passed, 3 skipped.
- Full default suite: suite5 1 unrelated TUI flake (test_main_getch_keyboard_interrupt_breaks, not reproducible in 27 targeted runs); suite6 2 failures (stale __pycache__ of test_devices.py serving an old _CNN with predict + marginal 0.1ms-over-threshold timing in test_quantization_benchmark); suite7 after : 0 failures, GREEN.

Notes: TUI getch KeyboardInterrupt flake (assert _running is False after sync _main) is logically impossible per static analysis (all break paths set _running=False); treated as environmental one-off under full-suite load. py_compile clean, pycache cleared.
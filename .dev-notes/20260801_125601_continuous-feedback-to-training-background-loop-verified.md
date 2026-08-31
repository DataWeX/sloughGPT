---
id: 20260801_125601_continuous-feedback-to-training-background-loop-verified
title: Continuous feedback-to-training background loop verified
status: done
tags: feedback,training,roadmap
created: 2026-08-01T12:56:01.944771+00:00
---

Continuous feedback-to-training background loop verified

Verified the continuous feedback-to-training background loop fires incrementally (not only on explicit aggregation).

1. Production wiring: workflow.start() runs in lifespan (main.py:137 via _start_feedback_workflow, gated by SLO_AUTO_WORKFLOW default true). scheduler_loop daemon thread calls _health_check -> run_scheduled_tasks every health_check_interval_seconds (30s), firing aggregate/prune/export/DPO on their interval timers.
2. Incremental per-feedback paths confirmed:
   - PerUserLoRAStore.update_adapter -> _auto_manage: auto-aggregates when quality_adapters >= auto_aggregate_threshold (50), auto-prunes when total_users >= auto_prune_threshold (100) - no explicit trigger_aggregate needed.
   - OnlineLoRAUpdater.add_feedback -> _trigger_update at update_interval threshold (existing test test_add_feedback_triggers_update_at_threshold).
   - FeedbackWorkflowManager.record_feedback pipes into meta weights, trait config, OnlineLoRAUpdater, PerUserLORAStore, and training data pipeline.
3. Added 7 tests in tests/test_feedback_domain.py covering: incremental auto-aggregate on update_adapter, incremental auto-prune on update_adapter, scheduler thread launch, health_check running scheduled tasks, loop incrementing workflow_runs, and record_feedback triggering incremental aggregation without explicit trigger.
4. Test results: tests/test_feedback_domain.py 105 passed, test_quality_guard 4 passed. test_workflow_router.py cannot collect (fastapi not installed - env limitation).
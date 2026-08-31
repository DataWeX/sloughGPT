---
id: 20260808_234705_audit-trail-instrumentation-for-privileged-operations
title: Audit-trail instrumentation for privileged operations
status: done
tags: security,audit,observability,sdl
created: 2026-08-08T23:47:05.151476+00:00
---

Audit-trail instrumentation for privileged operations

Instrumented privileged operations across 4 routers to emit structured audit events via AuditLogger.log (best-effort, try/except-wrapped so logging never breaks the operation).

Event types added:
- models.py: model.load (resource=model_id, extra={device,quantize}), model.unload (resource=current model id)
- souls.py: soul.switch (extra={checkpoint_name}), weights.snapshot.save/load/delete
- auto_train.py: training.start (resource=dataset_id|checkpoint|soul, extra={method,epochs}), training.checkpoint.delete, training.checkpoint.load
- datasets.py: dataset.delete (only on success; 404 path does not audit)

New helper audit_user(auth_user) in infrastructure/auth.py extracts sub/username/user_id or returns 'anonymous' (auth disabled default).

All events queryable at GET /security/audit?history=true&event_type=<type> (filter already supported by file_query + router).

Tests: tests/server/test_audit_instrumentation.py — 12 tests, all pass. Patch pattern: @patch('infrastructure.auth.get_audit_logger') + controller/domain mock per router. Includes audit-failure tolerance test (logger raising does not break POST /models/load).

Regression: 258 tests across security/souls/models/datasets/auto-train/auth suites pass. py_compile + ruff E9,F63,F7,F82 clean (2 pre-existing F823 in auto_train stream methods untouched). docs/API.md updated (audit instrumentation row in endpoint table + coverage row).
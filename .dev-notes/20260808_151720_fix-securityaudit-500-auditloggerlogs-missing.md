---
id: 20260808_151720_fix-securityaudit-500-auditloggerlogs-missing
title: Fix /security/audit 500 (AuditLogger.logs missing)
status: done
tags: security,audit,backend
created: 2026-08-08T15:17:20.490839+00:00
---

Fix /security/audit 500 (AuditLogger.logs missing)

Root cause: AuditLogger wrote JSON lines to rotating audit.log but exposed no .logs attribute, while GET /security/audit read audit_logger.logs[-limit:] -> AttributeError 500. Also record key was 'event' but router filters 'event_type'.

Fix (apps/api/server/infrastructure/auth.py):
- Added deque(maxlen=1000) ring buffer self._logs
- Added .logs property (newest last, returns a copy)
- log() now emits record key event_type (aligned with router + tests) and appends to buffer

Tests: 4 new real (unmocked) AuditLogger tests in tests/server/test_security_router.py (append w/ event_type, copy semantics, ring-buffer cap 1000, singleton). 34/34 security + 20/20 auth router pass. E2E verified: GET /security/audit 200 with filtered event_type. ruff E9/F63/F7/F82 clean.
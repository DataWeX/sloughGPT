---
id: 20260808_154352_persistent-audit-log-trail-history-cursor-pagination
title: Persistent audit-log trail (history + cursor pagination)
status: done
tags: security,audit,backend,frontend
created: 2026-08-08T15:43:52.241070+00:00
---

Persistent audit-log trail (history + cursor pagination)

Feature beyond roadmap: persistent audit-log trail.

Backend:
- AuditLogger.file_query(limit=100, event_type=None, before=None) in apps/api/server/infrastructure/auth.py — tails last 256KB of rotating audit.log, tolerant JSON-lines parse (skips non-{ / bad lines), newest-last, strict timestamp < before cursor, limit=0 -> all, negative mirrors logs[-limit:], OSError falls back to in-memory deque ring.
- GET /security/audit extended: history(bool, default False), before(str). history=true -> file_query, else in-memory ring as before.
- 10 new tests (TestAuditLoggerFileQuery): tail ordering, limit 0/positive/negative, event_type filter, before cursor, malformed-line skip, missing-file fallback, router history passthrough/before+filter/ring-ignores-before. Suite 44 security + 20 auth = 64 pass.

Frontend (apps/web/app/(app)/security/page.tsx):
- AuditLog interface aligned to real record (event_type/timestamp/user/resource/detail/extra); rows render resource/detail/extra instead of never-present ip/details.
- Card header: Session/Persisted toggle (history=true&limit=100), Load older (before=<oldest timestamp>, merge+dedup by timestamp|event_type), refresh respects current mode.
- Reverted useCallback([addToast]) effect loop (toast-store mock returns fresh addToast per selector call -> infinite reload); plain fetchData + useEffect([]) like original. Removed aria-labels that clobbered button accessible names. afterEach(cleanup) added to page test (vitest has no RTL auto-cleanup; DOM accumulates across tests in a file).
- 2 new page tests; 4/4 pass, tsc 0 errors.

Docs: docs/API.md Security history row added. Verified: py_compile + ruff E9/F63/F7/F82 clean; 64 backend tests + 4 web tests pass.
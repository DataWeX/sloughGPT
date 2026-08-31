---
id: 20260808_235515_wire-security-page-audit-filter-to-backend-event-type
title: Wire Security page audit filter to backend event_type
status: done
tags: security,audit,frontend,observability
created: 2026-08-08T23:55:15.602199+00:00
---

Wire Security page audit filter to backend event_type

Security page filter now passes event_type=<value> to the backend on every audit fetch, making pagination consistent within the filtered subset.

Changes (apps/web/app/(app)/security/page.tsx):
- Added eventParam() helper: returns '&event_type=<encodeURIComponent(filter)>' when filter is non-empty, '' otherwise (URLs unchanged when filter empty).
- fetchData(useHistory) appends eventParam() to the audit URL (initial + refresh + history toggle).
- loadOlder() appends eventParam() to the before-cursor URL, so Load older pages through the filtered subset instead of the unfiltered set.
- Client-side instant filter retained for typing feedback (no fetch-per-keystroke).

Tests (apps/web/app/(app)/security/page.test.tsx, +2 -> 6 total):
- 'passes event_type to backend when filter is set': types 'training', clicks refresh, asserts GET /security/audit?limit=100&event_type=training.
- 'encodes event_type and keeps it on load older': types 'training.start', toggles persisted, Load older asserts before cursor + &event_type=training.start.

Verification:
- 6/6 page tests pass, 8/8 SecurityOverviewCard tests pass.
- tsc --noEmit: 0 errors in security files (2 pre-existing errors in app/(app)/adapters/page.test.tsx, unrelated).
- Backend router already supported event_type filter on both ring and history paths (from prior increment), so no backend change required.
---
id: 20261113_153000_fix-registry-alias-calendar-http-client-bypass-dependency-ga
status: done
board: 20261113_153000_fix-registry-alias-calendar-http-client-bypass-dependency-ga
title: "fix: registry alias + calendar http-client bypass (dependency-gap fills 1+2)"
tags: ["api", "frontend", "tests"]
sprint: ""
gh: null
---

# fix: registry alias + calendar http-client bypass (dependency-gap fills 1+2)

Survey fills (user-approved, gaps 1+2 only): (1) E_CREATE_FAILED legacy alias at errors.py:112 shares E_BAD_REQUEST's enum identity so its registry write at :509 clobbers E_BAD_REQUEST's message ('Resource creation failed.' instead of 'Invalid request.'); alias is raised nowhere and unusable as a string code — remove alias+entry, restore message, add no-duplicate-values regression test (domain baseline rehash follows). (2) calendar/page.tsx:97 raw fetch('/api/calendar/events') bypasses http-client though url.ts SAME_ORIGIN_RE whitelists /api/calendar and planner-client uses apiGet — switch to apiGet. VERIFY: get_error_info('E_BAD_REQUEST') returns 'Invalid request.'; calendar page uses facade; error-handler + web gates green.

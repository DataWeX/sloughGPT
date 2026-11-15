---
id: 20261112_172008_fixtests-monitoring-page-test-mocks-uselivestatus-without-li
status: done
board: 20261112_172008_fixtests-monitoring-page-test-mocks-uselivestatus-without-li
title: "fix(tests): monitoring page test mocks useLiveStatus without liveStatusStore — unhandled rejection reddens Frontend unit tests on main"
tags: ["test", "frontend", "ci"]
sprint: ""
gh: null
---

# fix(tests): monitoring page test mocks useLiveStatus without liveStatusStore — unhandled rejection reddens Frontend unit tests on main

Symptom: 'Frontend unit tests' job exits 1 on main (runs 43238031014, 43222526186) and on PRs, despite all 7957 tests passing (787 test files pass; vitest reports 'Errors 1 error').

Root cause: apps/web/lib/api-monitor-store.ts:105 has a module-level side effect — import('@/hooks/useLiveStatus').then(({liveStatusStore}) => liveStatusStore.subscribe(...)). In jsdom window is defined, so it runs at import time. apps/web/app/(app)/monitoring/page.test.tsx mocks '@/hooks/useLiveStatus' but the mock body omits liveStatusStore, so the property access throws inside the uncaught .then() -> unhandled rejection -> exit 1.

Fix: add liveStatusStore to the monitoring test's vi.mock('@/hooks/useLiveStatus') factory, following the precedent in apps/web/app/(app)/training/page.test.tsx:
  liveStatusStore: { getState: () => ({health:null, connectionStatus:'connected'}), subscribe: () => () => {} }

Scope: frontend test mock only — no product code, no endpoint, no domain change. Own card + branch + PR (separate from PR #305 gates card).

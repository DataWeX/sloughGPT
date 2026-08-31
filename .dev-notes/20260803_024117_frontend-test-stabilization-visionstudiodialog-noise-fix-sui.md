---
id: 20260803_024117_frontend-test-stabilization-visionstudiodialog-noise-fix-sui
title: Frontend test stabilization: VisionStudioDialog noise fix, suite green
status: done
tags: frontend,tests,ui
created: 2026-08-03T02:41:17.352390+00:00
---

Frontend test stabilization: VisionStudioDialog noise fix, suite green

Full frontend suite green: 176 files / 1836 tests pass, tsc clean, 0 unhandled rejections. The roadmap '14 flaky frontend tests' item was already resolved (2026-06-22 blitz) — historically-flaky files (VisionStudioDialog, SoulSelectorDropdown, ChatMoreMenu) pass 5/5 under stress.

Fixed the only non-intentional test noise: VisionStudioDialog.test.tsx emitted a console.warn on every open=true render because the getTrainingReport mock returned undefined (no implementation), forcing the component into its catch path.
- getTrainingReport now defaults to a valid report (success path exercised, matching real API behavior)
- dev-log logger mocked with full method set (established pattern from ErrorLifecycle.test.tsx)
- 'shows initialCaps' test now explicitly models the offline/fallback scenario via mockRejectedValue
Remaining console messages are intentional error-path tests (useLocale throw assertion, model-controller list-failure log).
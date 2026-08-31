---
id: 20260809_000543_page-smoke-tests-and-suite-status
title: Page smoke tests for refactored pages; suite status + training/job/[id] in-flight
status: done
tags: frontend,tests,smoke,tsc
created: 2026-08-09T00:05:43.000000+00:00
---

Page smoke tests for refactored pages; suite status + training/job/[id] in-flight

Added page-level smoke tests for auth, benchmark, companion, datasets, errors, experiments, learn, tokenizer (apps/web/app/(app)/<page>/page.test.tsx). Fixes along the way: experiments list renders exp.id (not name); datasets/companion stable mocks (companion system-prompt card hidden when prompt empty); pre-existing tsc error in middleware.test.ts fixed via `as unknown as ReturnType<typeof mockHeaders>`. Full suite (apps/web): 847 suites / 2940 tests, 2923 passed. Remaining failures are all in training/job/[id]/ — page.test.tsx (19 tests, expects stat-Model/Active Training/Stop etc. not yet in page.tsx) and a scratch dbg.test.tsx (deleted by concurrent editor) — concurrent-editor in-flight work, left untouched. tsc --noEmit clean.
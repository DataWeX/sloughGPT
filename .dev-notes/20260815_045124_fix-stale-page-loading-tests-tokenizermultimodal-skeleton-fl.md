---
id: 20260815_045124_fix-stale-page-loading-tests-tokenizermultimodal-skeleton-fl
title: Fix stale page loading tests — tokenizer/multimodal skeleton flow
status: done
tags: frontend,web,tests,loading
created: 2026-08-15T04:51:24.462731+00:00
---

Fix stale page loading tests — tokenizer/multimodal skeleton flow

The tokenizer and multimodal page tests asserted page headers render synchronously, but both pages now load data via PageContainer with a loading prop, which renders PageSkeleton instead of the header while fetching. Updated 3 tests: tokenizer loading test asserts skeleton + no header then content after deferred getStats resolves; multimodal header test waits for load; multimodal loading test asserts header absent while pending then present after deferred capabilities resolve. Never-resolving-promise hacks replaced with deferred-promise resolution verifying the full skeleton->content transition. Full-suite failures were 3/3952; both files now 31/31 pass, tsc --noEmit clean.
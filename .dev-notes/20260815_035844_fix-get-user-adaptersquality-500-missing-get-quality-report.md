---
id: 20260815_035844_fix-get-user-adaptersquality-500-missing-get-quality-report
title: Fix GET /user-adapters/quality 500 — missing get_quality_report
status: done
tags: feedback,user-adapters,api
created: 2026-08-15T03:58:44.324331+00:00
---

Fix GET /user-adapters/quality 500 — missing get_quality_report

Root cause: router called store.get_quality_report() which did not exist on PerUserLoRAStore (only get_quality_adapters/get_stats). Fix: added get_quality_report(min_feedback_count=3, max_age_days=None) to packages/core-py/domains/feedback/per_user_lora.py returning {count, adapters} via get_quality_adapters; router get_quality now parses min_feedback_count/max_age_days query params (frontend user-adapters-controller getQuality sends these, expects {count, adapters}). Updated router-test mock lambda to accept **kw; added 5 store tests (empty, feedback filter, count consistency, age filter, max_age_days=0 no-op). 61 tests pass; real-store+router end-to-end smoke verified 200.
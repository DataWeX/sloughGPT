---
id: 20260809_001345_training-page-tests-job-detail-list
title: Training page tests (job detail + list)
status: done
tags: frontend,testing,training
created: 2026-08-09T00:13:45.062533+00:00
---

Training page tests (job detail + list)

Wrote Vitest + RTL tests for the two untested training routes in apps/web. job/[id]/page.test.tsx (19 tests): loading skeletons, completed/running/failed job views, KPIs, loss+reward with trend badge, stop/load-checkpoint/export/export-JSON/delete/cancel-delete/try-in-chat, refresh, polling. training/page.test.tsx (16 tests): header, stats derived from jobs+checkpoints, empty state, mount fetches, model status, dataset auto-select (search param + first), preview wiring, export metrics success/error, refresh, Ctrl+Enter start, canStart gate, tab switching via keyboard and click. Root cause found: unstable useRouter mock (new object per render) caused the page fetch effect (deps include router) to loop forever -> loading flicker races; fixed with a stable router object (mirrors DatasetDetailPage.test.tsx). Also fixed duplicate-text matches (job name appears in breadcrumb + title) via getAllByText, and corrected a trend badge assertion (page renders Math.abs(Number(pctChange)) = '50' not '50.0'). 35/35 pass, no new tsc errors (4 pre-existing errors in adapters/page.tsx, SelfTrainCard.tsx, self-train-controller.ts untouched).
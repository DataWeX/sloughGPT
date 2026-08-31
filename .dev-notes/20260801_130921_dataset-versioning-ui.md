---
id: 20260801_130921_dataset-versioning-ui
title: Dataset versioning UI
status: done
tags: datasets,ui
created: 2026-08-01T13:09:21.631943+00:00
---

Dataset versioning UI

Versioning UI added to dataset detail page (apps/web/app/(app)/dataset/[id]/page.tsx): Versions card with Create snapshot button, per-version Restore buttons, human-readable timestamp formatting, restore confirmation AlertDialog. Controller methods added in lib/dataset-controller.ts: createVersion POST /datasets/{id}/versions, listVersions GET /datasets/{id}/versions, restoreVersion POST /datasets/{id}/versions/{timestamp}. Fixed 9 failing page tests by adding version method mocks + IconClock to the @sloughgpt/strui mock; added 4 new page tests (empty state, list, create, restore flow). Added 3 controller tests + 6 backend router tests (run in CI where fastapi present). Verified: 15/15 DatasetDetailPage tests, 6/6 dataset-controller tests, full frontend 1829/1829, tsc clean.
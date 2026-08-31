---
id: 20260810_011622_dataset-flow-completion-searchversionsexport-tests-ci-green
title: Dataset flow completion: search/versions/export tests + CI green
status: done
tags: web,datasets,tests
created: 2026-08-10T01:16:22.925409+00:00
---

Dataset flow completion: search/versions/export tests + CI green

Dataset flow completed. Frontend: datasetController.search+export tests (5), DatasetsPage tests (13) covering search debounce indicator, version-count badges, export success/failure toasts. Backend bugfix: search_datasets returned bare name strings, but frontend renders full Dataset objects -> controller now delegates to list_datasets(q=q) returning full summaries (4 new controller tests). Fixed 2 stale export router tests (endpoint requires JSON body {format}, not query param; 422 -> 200). Page: searching indicator wired (was dead state). tsc exit 0; datasets surface green (24 web tests, 32 router tests). Full npm run ci NOT green this pass: 14 failures in TrainingSearch.test.tsx + WorkflowCard.test.tsx -- untracked files created mid-run (02:34/02:36) by a concurrent session editing them, unrelated to datasets.
---
id: 20260808_140501_server-router-test-thickening-datasets
title: Server router test thickening: datasets
status: done
tags: tests,server,datasets
created: 2026-08-08T14:05:01.668656+00:00
---

Server router test thickening: datasets

Thickened test_datasets_router.py 30 -> 76 tests (+46). Gaps closed: github/huggingface/url/isbn import success+failure via patched data_import classes; import schema 422s (missing url/name/isbn/kaggle/csv); batch import mixed/unsupported/local/github-failure/empty; search 422 bounds (missing/empty/501-char q, books/github limit 0/51); from-chat empty-messages 400 + missing-messages/bad-role/name-too-long 422s; convert-to-messages 404 paths + full text+messages conversion (dataset_id is a QUERY param, not path — route registeration has no {dataset_id} segment); export format case-sensitive 422 (CSV) + csv success; preview limit bounds; list q/type passthrough + count; create_dataset 422s + full fixture. Full server suite green: 1825 passed, 68 deselected (was 1779). Also re-verified tokenizer/agents/vm suites from prior batches remain green.
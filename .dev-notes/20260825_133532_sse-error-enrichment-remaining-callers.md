---
id: 20260825_133532_sse-error-enrichment-remaining-callers
title: SSE Error Enrichment - Remaining Callers
status: done
tags: area,backend
created: 2026-08-25T13:35:32.015223+00:00
---

SSE Error Enrichment - Remaining Callers

Enriched all remaining sse_error callers across auto_train.py, mobile.py, agents.py with structured codes (E_STATE_IDLE, E_TIMEOUT, E_VAL_REQUEST, E_ENV_MISSING, E_INFRA_GENERATION). Updated 3 fallback definitions to accept code/http_status params. 15 calls enriched. All tests pass.
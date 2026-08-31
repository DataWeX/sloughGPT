---
id: 20260824_112504_backend-error-classification-health
title: Backend error classification & health
status: done
tags: backend
created: 2026-08-24T11:25:04.474145+00:00
---

Backend error classification & health

Fixed souls.py list_souls/list_weight_snapshots to use classify_and_raise instead of raise_error(str(e)). Fixed health.py with try/except and proper error messages.
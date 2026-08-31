---
id: 20260824_172917_fixed-nameerror-in-dataset-import-error-handlers
title: Fixed NameError in dataset import error handlers
status: done
tags: backend
created: 2026-08-24T17:29:17.617949+00:00
---

Fixed NameError in dataset import error handlers

Changed all except HTTPException: classify_and_raise(e) to except HTTPException: raise in datasets.py import handlers. Fixed NameError where e was undefined.
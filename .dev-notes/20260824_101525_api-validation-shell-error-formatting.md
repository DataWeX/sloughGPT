---
id: 20260824_101525_api-validation-shell-error-formatting
title: API validation & shell error formatting
status: done
tags: shell,backend
created: 2026-08-24T10:15:25.571844+00:00
---

API validation & shell error formatting

Added Field constraints to images.py GenerateRequest and souls.py SloChatRequest. Added _format_error helper for descriptive error messages with command context and actionable hints. Updated execute/execute_single/background error handlers.
---
id: 20260808_022006_full-suite-flake-models-422-root-cause-fixed
title: Full-suite flake: /models 422 root cause fixed
status: done
tags: server,tests,models
created: 2026-08-08T02:20:06.313879+00:00
---

Full-suite flake: /models 422 root cause fixed

Root cause: ModelsController.get_current_model() returned a dict with device=None whenever _current_model was set without _current_device (process-guard path). routers/models.py list_models then built ModelInfo(device=None) -> pydantic string_type validation error -> 422 'Validation failed on /models' fields=1, intermittently failing test_server_api/test_list_models + test_model_has_required_fields in the full suite (then cascading to inference no-model tests). Fix: get_current_model() returns None unless BOTH model and device are set (apps/api/server/controllers/models.py:744). Server suite: 819 passed, 68 deselected, stable across 3 runs.
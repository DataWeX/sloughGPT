---
id: 20260805_081814_fine-tuned-load-now-unregisters-stale-registry-default
title: Fine-tuned load now unregisters stale registry default
status: done
tags: api,model-registry,bugfix
created: 2026-08-05T08:18:14.362004+00:00
---

Fine-tuned load now unregisters stale registry default

Fixed /health inconsistency after loading a local fine-tuned model via load_model_path/load_finetuned_model. load_model_path set controller + provider but never touched ModelRegistry, so health.py (registry-first) kept reporting the stale autoloaded HF default (e.g. Qwen) while chat actually used the fine-tuned SLoNet provider. Now on successful load it unregisters registry.default_id when it differs from the loaded model_id. Added 2 tests (stale default unregistered, matching default kept). 9 suite tests pass, 43 backend, provider suite pass, py_compile clean.
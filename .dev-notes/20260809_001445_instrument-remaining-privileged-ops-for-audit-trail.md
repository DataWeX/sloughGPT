---
id: 20260809_001445_instrument-remaining-privileged-ops-for-audit-trail
title: Instrument remaining privileged ops for audit trail
status: done
tags: security,audit,backend
created: 2026-08-09T00:14:45.817297+00:00
---

Instrument remaining privileged ops for audit trail

Extended AuditLogger instrumentation to the remaining privileged operations:
- routers/models.py: model.quantize, model.dequantize, model.precision, model.download, model.cancel (new _audit_model_id helper; handlers gained auth_user dependency)
- training/router.py: training.start (detail char/hf), training.stop, training.delete, training.webhook.register/delete
- routers/souls.py: soul.weights.save
- tests/server/test_audit_instrumentation.py: 12 -> 27 tests, all pass
- Fixed pre-existing F821 in training/router.py quick_train (cancel_event kwarg never supported by HFFineTuner/GRPOTrainer.train -> dropped)
- docs/API.md audit instrumentation row updated
- Verification: py_compile, ruff (E9/F63/F7/F82) clean, tests/server full suite 1871 passed

Session 2 (this note continues):
- Added 41 new audit tests across routers: kb.py, agents.py, multimodal.py, config.py, experiments.py, user_adapters.py, lora_eval.py, tokenizer.py, system.py, self_train.py, auto_train.py (pause/resume/cancel_from_sessions), datasets.py (create/update/version/append/convert) -> test_audit_instrumentation.py now 68 tests
- Verified audit call sites: kb add/update/batch_ingest/batch_delete/delete; agents create/update/delete/execute; config.generation.save (logs even on empty updates); experiment.create/delete; adapter.update/reset/merge/aggregate/delete/prune; adapter.eval.aggregate (resource=output_name); tokenizer.train; executor.purge/cancel; self_train.start/stop; training.pause/resume; training.stop (cancel_from_sessions, GET); dataset.create/update/version/append/convert
- Debugged failures during extension: VersionCreateResponse.timestamp is str (mock must return string not int); patched module attr _instance must carry return_value directly (purge/cancel); experiments delete shares one mock with create (assert last call)
- docs/API.md audit table row + coverage list expanded
- Verification: py_compile + ruff clean; tests/server full suite 1912 passed, 68 deselected, 1 warning